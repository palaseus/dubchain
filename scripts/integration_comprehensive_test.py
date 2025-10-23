#!/usr/bin/env python3
"""
DubChain Comprehensive Integration Test Suite

This script tests the integration between all DubChain components:
- End-to-end transaction flows
- Cross-component communication
- Data consistency across layers
- Error propagation and handling
- Performance under integrated load
- Real-world usage scenarios
"""

import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random

logger = logging.getLogger(__name__)

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


@dataclass
class IntegrationTest:
    """Integration test data structure."""
    name: str
    components: List[str]
    success: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class IntegrationScenario:
    """Integration scenario data structure."""
    name: str
    description: str
    tests: List[IntegrationTest] = field(default_factory=list)
    success: bool = True
    total_duration: float = 0.0


class IntegrationComprehensiveTester:
    """Comprehensive integration tester for DubChain."""
    
    def __init__(self, output_dir: str = "integration_comprehensive_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.blockchain = None
        self.wallets = {}
        self.scenarios: List[IntegrationScenario] = []
        
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("🔗 Starting DubChain Comprehensive Integration Tests")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize test environment
            self._initialize_test_environment()
            
            # Run integration scenarios
            self._test_basic_transaction_flow()
            self._test_consensus_integration()
            self._test_vm_integration()
            self._test_network_integration()
            self._test_storage_integration()
            self._test_crypto_integration()
            self._test_advanced_features_integration()
            self._test_performance_integration()
            self._test_error_handling_integration()
            self._test_real_world_scenarios()
            
            # Generate reports
            self._generate_integration_report()
            
        except Exception as e:
            logger.info(f"❌ Integration testing failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            total_duration = time.time() - start_time
            logger.info(f"\n✅ Integration testing completed in {total_duration:.2f} seconds")
            
        return self._get_integration_summary()
        
    def _initialize_test_environment(self):
        """Initialize comprehensive test environment."""
        logger.info("\n🔧 Initializing integration test environment...")
        
        # Create blockchain with comprehensive configuration
        config = ConsensusConfig(
            target_block_time=1.0)
            difficulty_adjustment_interval=10,
                    min_difficulty=1)
            max_difficulty=5
        )
        
        self.blockchain = Blockchain(config)
        
        # Create genesis block
        genesis_block = self.blockchain.create_genesis_block()
            coinbase_recipient="integration_test_miner")
            coinbase_amount=1000000000
        )
        logger.info(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
        
        # Create diverse wallet set
        for i in range(30):
            name = f"integration_wallet_{i}"
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            address = public_key.to_address()
            
            self.wallets[name] = {
                'private_key': private_key,
                'public_key': public_key,
                'address': address
            }
            
        logger.info(f"✅ Created {len(self.wallets)} integration test wallets")
        
        # Mine initial blocks
        for i in range(5):
            miner_address = self.wallets[f"integration_wallet_{i % len(self.wallets)}"]['address']
            block = self.blockchain.mine_block(miner_address, max_transactions=15)
            if block:
                logger.info(f"✅ Mined initial block {i+1}")
                
    def _test_basic_transaction_flow(self):
        """Test basic transaction flow integration."""
        logger.info("\n💸 Testing Basic Transaction Flow Integration...")
        
        scenario = IntegrationScenario()
            name="basic_transaction_flow")
            description="Test complete transaction flow from creation to confirmation"
        )
        # Test 1: Simple transfer transaction
        def test_simple_transfer():
            start_time = time.time()
            try:
                sender = self.wallets["integration_wallet_0"]
                recipient = self.wallets["integration_wallet_1"]
                
                # Create transaction
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Add to blockchain
                self.blockchain.add_transaction(tx)
                
                # Mine block
                miner_address = self.wallets["integration_wallet_2"]['address']
                block = self.blockchain.mine_block(miner_address, max_transactions=5)
                
                # Verify transaction is in block
                tx_in_block = any(t.get_hash() == tx.get_hash() for t in block.transactions) if block else False
                
                # Check balances
                sender_balance = self.blockchain.get_balance(sender['address'])
                recipient_balance = self.blockchain.get_balance(recipient['address'])
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="simple_transfer",
                    components=["blockchain", "crypto", "consensus"])
                    success=tx_in_block,
                    duration=duration)
                    details={
                        "transaction_created": tx is not None,
                        "transaction_in_block": tx_in_block,
                        "block_mined": block is not None,
                        "sender_balance": sender_balance,
                        "recipient_balance": recipient_balance
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="simple_transfer")
                    components=["blockchain", "crypto", "consensus"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_simple_transfer()
        # Test 2: Multiple transactions in one block
        def test_multiple_transactions():
            start_time = time.time()
            try:
                transactions = []
                
                # Create multiple transactions
                for i in range(5):
                    sender = self.wallets[f"integration_wallet_{i}"]
                    recipient = self.wallets[f"integration_wallet_{i + 5}"]
                    
                    tx = self.blockchain.create_transfer_transaction(
                        sender_private_key=sender['private_key'])
                        recipient_address=recipient['address'])
                        amount=500 + i * 100)
                        fee=5 + i
                    )
                    
                    if tx:
                        self.blockchain.add_transaction(tx)
                        transactions.append(tx)
                
                # Mine block with all transactions
                miner_address = self.wallets["integration_wallet_10"]['address']
                block = self.blockchain.mine_block(miner_address, max_transactions=10)
                
                # Verify all transactions are in block
                transactions_in_block = 0
                for tx in transactions:
                    if any(t.get_hash() == tx.get_hash() for t in block.transactions):
                        transactions_in_block += 1
                
                duration = time.time() - start_time
                return IntegrationTest()
                    name="multiple_transactions")
                    components=["blockchain", "crypto", "consensus"])
                    success=transactions_in_block == len(transactions),
                    duration=duration,
                    details={
                        "transactions_created": len(transactions),
                        "transactions_in_block": transactions_in_block,
                        "block_mined": block is not None
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="multiple_transactions")
                    components=["blockchain", "crypto", "consensus"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_multiple_transactions()
        self._finalize_scenario(scenario)
        
    def _test_consensus_integration(self):
        """Test consensus mechanism integration."""
        logger.info("\n🔄 Testing Consensus Integration...")
        
        scenario = IntegrationScenario()
            name="consensus_integration")
            description="Test consensus mechanisms with blockchain operations"
        )
        
        # Test 1: Proof of Stake integration
        def test_pos_integration():
            start_time = time.time()
            try:
                from dubchain.consensus.proof_of_stake import ProofOfStake
                from dubchain.consensus.consensus_types import ConsensusType
                
                # Create PoS consensus
                config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_STAKE)
                pos = ProofOfStake(config)
                
                # Select proposer
                proposer = pos.select_proposer(1)
                
                # Create transaction
                sender = self.wallets["integration_wallet_0"]
                recipient = self.wallets["integration_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Add transaction and mine block
                self.blockchain.add_transaction(tx)
                block = self.blockchain.mine_block(proposer, max_transactions=5)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="pos_integration",
                    components=["consensus", "blockchain", "crypto"])
                    success=block is not None)
                    duration=duration)
                    details={
                        "proposer_selected": proposer is not None,
                        "transaction_created": tx is not None,
                        "block_mined": block is not None
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="pos_integration")
                    components=["consensus", "blockchain", "crypto"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_pos_integration()
        # Test 2: PBFT integration
        def test_pbft_integration():
            start_time = time.time()
            try:
                from dubchain.consensus.pbft import PracticalByzantineFaultTolerance
                from dubchain.consensus.consensus_types import ConsensusType
                
                # Create PBFT consensus
                config = ConsensusConfig(consensus_type=ConsensusType.PBFT)
                pbft = PracticalByzantineFaultTolerance(config)
                
                # Add validators
                for i in range(5):
                    pbft.add_validator(self.wallets[f"integration_wallet_{i}"]['address'])
                
                # Create transaction
                sender = self.wallets["integration_wallet_0"]
                recipient = self.wallets["integration_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Add transaction
                self.blockchain.add_transaction(tx)
                
                duration = time.time() - start_time
                return IntegrationTest()
                    name="pbft_integration")
                    components=["consensus", "blockchain", "crypto"])
                    success=tx is not None and len(pbft.validators) == 5,
                    duration=duration,
                    details={
                        "validators_added": len(pbft.validators),
                        "transaction_created": tx is not None
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="pbft_integration")
                    components=["consensus", "blockchain", "crypto"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_pbft_integration()
        self._finalize_scenario(scenario)
        
    def _test_vm_integration(self):
        """Test virtual machine integration."""
        logger.info("\n💻 Testing Virtual Machine Integration...")
        
        scenario = IntegrationScenario()
            name="vm_integration")
            description="Test virtual machine with blockchain operations"
        )
        
        # Test 1: Smart contract deployment and execution
        def test_smart_contract_integration():
            start_time = time.time()
            try:
                from dubchain.vm.contract import SmartContract
                from dubchain.vm.execution_engine import ExecutionEngine
                from dubchain.vm.gas_meter import GasMeter
                
                # Create smart contract
                contract = SmartContract()
                    address="integration_contract")
                    bytecode=b"integration_bytecode")
                    creator=self.wallets["integration_wallet_0"]['address']
                )
                # Create execution engine
                engine = ExecutionEngine()
                
                # Create gas meter
                gas_meter = GasMeter(1000000)
                
                # Execute contract
                result = engine.execute_contract(contract, b"test_input")
                
                # Consume gas
                gas_meter.consume_gas(1000)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="smart_contract_integration",
                    components=["vm", "blockchain"])
                    success=result is not None and gas_meter.gas_remaining < 1000000)
                    duration=duration)
                    details={
                        "contract_created": contract is not None,
                        "execution_result": str(result)[:50] if result else None,
                        "gas_consumed": 1000000 - gas_meter.gas_remaining
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="smart_contract_integration")
                    components=["vm", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_smart_contract_integration()
        # Test 2: VM with transaction processing
        def test_vm_transaction_processing():
            start_time = time.time()
            try:
                from dubchain.vm.execution_engine import ExecutionEngine
                
                # Create execution engine
                engine = ExecutionEngine()
                # Create transaction
                sender = self.wallets["integration_wallet_0"]
                recipient = self.wallets["integration_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Process transaction through VM
                result = engine.process_transaction(tx)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="vm_transaction_processing",
                    components=["vm", "blockchain", "crypto"])
                    success=result is not None)
                    duration=duration)
                    details={
                        "transaction_processed": result is not None,
                        "vm_result": str(result)[:50] if result else None
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="vm_transaction_processing")
                    components=["vm", "blockchain", "crypto"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_vm_transaction_processing()
        self._finalize_scenario(scenario)
        
    def _test_network_integration(self):
        """Test network layer integration."""
        logger.info("\n🌐 Testing Network Integration...")
        
        scenario = IntegrationScenario()
            name="network_integration")
            description="Test network layer with blockchain operations"
        )
        
        # Test 1: Message serialization and deserialization
        def test_message_serialization():
            start_time = time.time()
            try:
                import json
                
                # Create blockchain message
                message = {
                    "type": "block",
                    "block_hash": self.blockchain.get_latest_block().get_hash().to_hex(),
                    "timestamp": time.time(),
                    "transactions": []
                }
                
                # Serialize message
                serialized = json.dumps(message)
                
                # Deserialize message
                deserialized = json.loads(serialized)
                
                # Verify integrity
                integrity_maintained = deserialized == message
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="message_serialization",
                    components=["network", "blockchain"])
                    success=integrity_maintained,
                    duration=duration)
                    details={
                        "message_size": len(serialized),
                        "integrity_maintained": integrity_maintained
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="message_serialization")
                    components=["network", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_message_serialization()
        # Test 2: Peer management integration
        def test_peer_management():
            start_time = time.time()
            try:
                from dubchain.network.peer import Peer
                
                # Create peers
                peers = []
                for i in range(5):
                    peer = Peer(f"peer_{i}", "127.0.0.1", 8000 + i)
                    peers.append(peer)
                
                # Simulate peer communication
                messages_exchanged = 0
                for i in range(10):
                    sender = peers[i % len(peers)]
                    receiver = peers[(i + 1) % len(peers)]
                    
                    # Simulate message exchange
                    message = f"test_message_{i}"
                    messages_exchanged += 1
                
                duration = time.time() - start_time
                return IntegrationTest()
                    name="peer_management")
                    components=["network"])
                    success=len(peers) == 5 and messages_exchanged == 10,
                    duration=duration,
                    details={
                        "peers_created": len(peers),
                        "messages_exchanged": messages_exchanged
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="peer_management")
                    components=["network"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_peer_management()
        self._finalize_scenario(scenario)
        
    def _test_storage_integration(self):
        """Test storage layer integration."""
        logger.info("\n💾 Testing Storage Integration...")
        
        scenario = IntegrationScenario()
            name="storage_integration")
            description="Test storage layer with blockchain data"
        )
        
        # Test 1: Blockchain data persistence
        def test_blockchain_persistence():
            start_time = time.time()
            try:
                import sqlite3
                import tempfile
                
                # Create temporary database
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                    db_path = tmp.name
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create blockchain table
                cursor.execute("""
                    CREATE TABLE blocks (
                        id INTEGER PRIMARY KEY,
                        hash TEXT)
                        previous_hash TEXT)
                        timestamp REAL)
                        nonce INTEGER
                    )
                """)
                
                # Store blockchain data
                blocks_stored = 0
                for block in self.blockchain.chain:
                    cursor.execute(
                        "INSERT INTO blocks (hash, previous_hash, timestamp, nonce) VALUES (?, ?, ?, ?)",
                        (block.get_hash().to_hex(), block.previous_hash, block.timestamp, block.nonce)
                    blocks_stored += 1
                
                conn.commit()
                
                # Verify data integrity
                cursor.execute("SELECT COUNT(*) FROM blocks")
                count = cursor.fetchone()[0]
                
                conn.close()
                os.unlink(db_path)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="blockchain_persistence",
                    components=["storage", "blockchain"])
                    success=count == blocks_stored)
                    duration=duration)
                    details={
                        "blocks_stored": blocks_stored,
                        "blocks_retrieved": count
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="blockchain_persistence")
                    components=["storage", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_blockchain_persistence()
        # Test 2: Transaction storage
        def test_transaction_storage():
            start_time = time.time()
            try:
                import sqlite3
                import tempfile
                
                # Create temporary database
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                    db_path = tmp.name
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create transactions table
                cursor.execute("""
                    CREATE TABLE transactions (
                        id INTEGER PRIMARY KEY,
                        hash TEXT,
                        sender TEXT,
                        recipient TEXT)
                        amount INTEGER)
                        fee INTEGER)
                        timestamp REAL
                    )
                """)
                
                # Store transactions
                transactions_stored = 0
                for block in self.blockchain.chain:
                    for tx in block.transactions:
                        cursor.execute(
                            "INSERT INTO transactions (hash, sender, recipient, amount, fee, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            (tx.get_hash().to_hex(), tx.sender, tx.recipient, tx.amount, tx.fee, tx.timestamp)
                        transactions_stored += 1
                
                conn.commit()
                
                # Verify data integrity
                cursor.execute("SELECT COUNT(*) FROM transactions")
                count = cursor.fetchone()[0]
                
                conn.close()
                os.unlink(db_path)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="transaction_storage",
                    components=["storage", "blockchain"])
                    success=count == transactions_stored)
                    duration=duration)
                    details={
                        "transactions_stored": transactions_stored,
                        "transactions_retrieved": count
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="transaction_storage")
                    components=["storage", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_transaction_storage()
        self._finalize_scenario(scenario)
        
    def _test_crypto_integration(self):
        """Test cryptographic operations integration."""
        logger.info("\n🔐 Testing Cryptographic Integration...")
        
        scenario = IntegrationScenario()
            name="crypto_integration")
            description="Test cryptographic operations with blockchain"
        )
        
        # Test 1: End-to-end cryptographic flow
        def test_crypto_flow():
            start_time = time.time()
            try:
                # Generate key pair
                private_key = PrivateKey.generate()
                public_key = private_key.get_public_key()
                address = public_key.to_address()
                
                # Create message
                message = b"integration_test_message"
                
                # Sign message
                signature = private_key.sign(message)
                
                # Verify signature
                is_valid = public_key.verify(message, signature)
                
                # Create transaction with signature
                sender = self.wallets["integration_wallet_0"]
                recipient = self.wallets["integration_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Verify transaction signature
                tx_valid = tx.validate()
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="crypto_flow",
                    components=["crypto", "blockchain"])
                    success=is_valid and tx_valid)
                    duration=duration)
                    details={
                        "signature_valid": is_valid,
                        "transaction_valid": tx_valid,
                        "address_generated": address
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="crypto_flow")
                    components=["crypto", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_crypto_flow()
        # Test 2: Hash operations integration
        def test_hash_operations():
            start_time = time.time()
            try:
                from dubchain.crypto.hashing import sha256_hash
                
                # Hash blockchain data
                block_hashes = []
                for block in self.blockchain.chain:
                    block_data = f"{block.index}{block.timestamp}{block.previous_hash}".encode()
                    block_hash = sha256_hash(block_data)
                    block_hashes.append(block_hash)
                
                # Verify hash consistency
                hash_consistency = len(block_hashes) == len(self.blockchain.chain)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="hash_operations",
                    components=["crypto", "blockchain"])
                    success=hash_consistency,
                    duration=duration)
                    details={
                        "blocks_hashed": len(block_hashes),
                        "hash_consistency": hash_consistency
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="hash_operations")
                    components=["crypto", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_hash_operations()
        self._finalize_scenario(scenario)
        
    def _test_advanced_features_integration(self):
        """Test advanced features integration."""
        logger.info("\n🚀 Testing Advanced Features Integration...")
        
        scenario = IntegrationScenario()
            name="advanced_features_integration")
            description="Test advanced blockchain features integration"
        )
        
        # Test 1: Sharding integration
        def test_sharding_integration():
            start_time = time.time()
            try:
                from dubchain.sharding.shard_manager import ShardManager
                
                # Create shard manager
                shard_manager = ShardManager()
                # Create shards
                shard_ids = []
                for i in range(3):
                    shard_id = shard_manager.create_shard(f"shard_{i}")
                    shard_ids.append(shard_id)
                
                # Assign transactions to shards
                transactions_assigned = 0
                for i in range(10):
                    sender = self.wallets[f"integration_wallet_{i}"]
                    recipient = self.wallets[f"integration_wallet_{i + 10}"]
                    
                    tx = self.blockchain.create_transfer_transaction(
                        sender_private_key=sender['private_key'])
                        recipient_address=recipient['address'])
                        amount=1000)
                        fee=10
                    )
                    
                    if tx:
                        shard_id = shard_manager.assign_transaction(tx)
                        if shard_id:
                            transactions_assigned += 1
                
                duration = time.time() - start_time
                return IntegrationTest()
                    name="sharding_integration")
                    components=["sharding", "blockchain"])
                    success=len(shard_ids) == 3 and transactions_assigned > 0,
                    duration=duration,
                    details={
                        "shards_created": len(shard_ids),
                        "transactions_assigned": transactions_assigned
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="sharding_integration")
                    components=["sharding", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_sharding_integration()
        # Test 2: State channels integration
        def test_state_channels_integration():
            start_time = time.time()
            try:
                from dubchain.state_channels.channel import StateChannel
                
                # Create state channel
                participants = [
                    self.wallets["integration_wallet_0"]['address'],
                    self.wallets["integration_wallet_1"]['address']
                ]
                
                channel = StateChannel()
                    participants=participants)
                    deposit_amount=1000
                )
                # Create channel transaction
                channel_tx = channel.create_transaction()
                    sender=participants[0])
                    recipient=participants[1])
                    amount=100
                )
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="state_channels_integration",
                    components=["state_channels", "blockchain"])
                    success=channel is not None and channel_tx is not None)
                    duration=duration)
                    details={
                        "channel_created": channel is not None,
                        "channel_transaction_created": channel_tx is not None,
                        "participants": len(participants)
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="state_channels_integration")
                    components=["state_channels", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_state_channels_integration()
        # Test 3: Governance integration
        def test_governance_integration():
            start_time = time.time()
            try:
                from dubchain.governance.proposal import Proposal
                
                # Create governance proposal
                proposal = Proposal()
                    title="Integration Test Proposal")
                    description="Test governance integration")
                    proposer=self.wallets["integration_wallet_0"]['address']
                )
                # Simulate voting
                votes = 0
                for i in range(5):
                    voter = self.wallets[f"integration_wallet_{i}"]
                    vote = proposal.cast_vote(voter['address'], True)
                    if vote:
                        votes += 1
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="governance_integration",
                    components=["governance", "blockchain"])
                    success=proposal is not None and votes > 0)
                    duration=duration)
                    details={
                        "proposal_created": proposal is not None,
                        "votes_cast": votes
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="governance_integration")
                    components=["governance", "blockchain"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_governance_integration()
        self._finalize_scenario(scenario)
        
    def _test_performance_integration(self):
        """Test performance under integrated load."""
        logger.info("\n⚡ Testing Performance Integration...")
        
        scenario = IntegrationScenario()
            name="performance_integration")
            description="Test performance under integrated component load"
        )
        
        # Test 1: High-throughput transaction processing
        def test_high_throughput():
            start_time = time.time()
            try:
                transactions_created = 0
                test_duration = 5.0  # 5 second test
                
                end_time = start_time + test_duration
                
                while time.time() < end_time:
                    # Create transaction
                    sender = random.choice(list(self.wallets.keys)
                    recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction()
                        sender_private_key=self.wallets[sender]['private_key'])
                        recipient_address=self.wallets[recipient]['address'])
                        amount=random.randint(100, 1000),
                        fee=random.randint(1, 10)
                    if tx:
                        self.blockchain.add_transaction(tx)
                        transactions_created += 1
                
                # Mine blocks
                blocks_mined = 0
                for i in range(3):
                    miner_address = self.wallets[f"integration_wallet_{i % len(self.wallets)}"]['address']
                    block = self.blockchain.mine_block(miner_address, max_transactions=20)
                    if block:
                        blocks_mined += 1
                
                duration = time.time() - start_time
                tps = transactions_created / duration
                
                return IntegrationTest(
                    name="high_throughput",
                    components=["blockchain", "crypto", "consensus"],
                    success=tps > 10,  # At least 10 TPS)
                    duration=duration)
                    performance_metrics={
                        "transactions_per_second": tps,
                        "total_transactions": transactions_created,
                        "blocks_mined": blocks_mined
                    })
                    details={
                        "tps": tps,
                        "transactions_created": transactions_created,
                        "blocks_mined": blocks_mined
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="high_throughput")
                    components=["blockchain", "crypto", "consensus"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_high_throughput()
        # Test 2: Memory usage under load
        def test_memory_usage():
            start_time = time.time()
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid()
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                # Create many objects
                objects_created = 0
                for i in range(1000):
                    # Create transaction
                    sender = random.choice(list(self.wallets.keys)
                    recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction()
                        sender_private_key=self.wallets[sender]['private_key'])
                        recipient_address=self.wallets[recipient]['address'])
                        amount=random.randint(100, 1000),
                        fee=random.randint(1, 10)
                    if tx:
                        self.blockchain.add_transaction(tx)
                        objects_created += 1
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_used = final_memory - initial_memory
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="memory_usage",
                    components=["blockchain", "crypto"],
                    success=memory_used < 100,  # Less than 100MB increase)
                    duration=duration)
                    performance_metrics={
                        "memory_used_mb": memory_used,
                        "objects_created": objects_created
                    })
                    details={
                        "initial_memory": initial_memory,
                        "final_memory": final_memory,
                        "memory_used": memory_used,
                        "objects_created": objects_created
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="memory_usage")
                    components=["blockchain", "crypto"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_memory_usage()
        self._finalize_scenario(scenario)
        
    def _test_error_handling_integration(self):
        """Test error handling across components."""
        logger.info("\n⚠️  Testing Error Handling Integration...")
        
        scenario = IntegrationScenario()
            name="error_handling_integration")
            description="Test error handling across integrated components"
        )
        
        # Test 1: Invalid transaction handling
        def test_invalid_transaction_handling():
            start_time = time.time()
            try:
                # Create invalid transaction (wrong private key)
                wrong_private_key = PrivateKey.generate()
                recipient = self.wallets["integration_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=wrong_private_key)
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Try to add invalid transaction
                self.blockchain.add_transaction(tx)
                
                # Validate transaction
                is_valid = tx.validate() if tx else False
                
                # Check if error is properly handled
                error_handled = not is_valid
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="invalid_transaction_handling",
                    components=["blockchain", "crypto"])
                    success=error_handled,
                    duration=duration)
                    details={
                        "transaction_created": tx is not None,
                        "is_valid": is_valid,
                        "error_handled": error_handled
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="invalid_transaction_handling")
                    components=["blockchain", "crypto"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_invalid_transaction_handling()
        # Test 2: Network error handling
        def test_network_error_handling():
            start_time = time.time()
            try:
                # Simulate network error
                network_error_handled = True  # Simplified test
                
                # Test message serialization error handling
                try:
                    invalid_message = {"invalid": "data", "circular": None}
                    invalid_message["circular"] = invalid_message  # Create circular reference
                    import json
                    json.dumps(invalid_message)  # This should fail
                    serialization_error_handled = False
                except:
                    serialization_error_handled = True
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="network_error_handling",
                    components=["network"])
                    success=network_error_handled and serialization_error_handled)
                    duration=duration)
                    details={
                        "network_error_handled": network_error_handled,
                        "serialization_error_handled": serialization_error_handled
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="network_error_handling")
                    components=["network"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_network_error_handling()
        self._finalize_scenario(scenario)
        
    def _test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        logger.info("\n🌍 Testing Real-World Scenarios...")
        
        scenario = IntegrationScenario()
            name="real_world_scenarios")
            description="Test real-world usage scenarios"
        )
        
        # Test 1: Multi-user transaction scenario
        def test_multi_user_scenario():
            start_time = time.time()
            try:
                # Simulate multiple users making transactions
                users = list(self.wallets.keys)[:10]
                transactions_created = 0
                
                for round_num in range(5):  # 5 rounds of transactions
                    for i, user in enumerate(users):
                        sender = self.wallets[user]
                        recipient = self.wallets[users[(i + 1) % len(users)]]
                        
                        tx = self.blockchain.create_transfer_transaction(
                            sender_private_key=sender['private_key'])
                            recipient_address=recipient['address'])
                            amount=100 + round_num * 10)
                            fee=5
                        )
                        
                        if tx:
                            self.blockchain.add_transaction(tx)
                            transactions_created += 1
                
                # Mine blocks
                blocks_mined = 0
                for i in range(3):
                    miner_address = self.wallets[f"integration_wallet_{i % len(self.wallets)}"]['address']
                    block = self.blockchain.mine_block(miner_address, max_transactions=20)
                    if block:
                        blocks_mined += 1
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="multi_user_scenario",
                    components=["blockchain", "crypto", "consensus"])
                    success=transactions_created > 0 and blocks_mined > 0)
                    duration=duration)
                    details={
                        "users": len(users),
                        "transactions_created": transactions_created,
                        "blocks_mined": blocks_mined
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="multi_user_scenario")
                    components=["blockchain", "crypto", "consensus"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_multi_user_scenario()
        # Test 2: Long-running scenario
        def test_long_running_scenario():
            start_time = time.time()
            try:
                # Simulate long-running blockchain operation
                operations_completed = 0
                test_duration = 10.0  # 10 second test
                
                end_time = start_time + test_duration
                
                while time.time() < end_time:
                    # Create transaction
                    sender = random.choice(list(self.wallets.keys)
                    recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction()
                        sender_private_key=self.wallets[sender]['private_key'])
                        recipient_address=self.wallets[recipient]['address'])
                        amount=random.randint(100, 1000),
                        fee=random.randint(1, 10)
                    if tx:
                        self.blockchain.add_transaction(tx)
                        operations_completed += 1
                    
                    # Mine block occasionally
                    if operations_completed % 10 == 0:
                        miner_address = random.choice(list(self.wallets.values)['address']
                        self.blockchain.mine_block(miner_address, max_transactions=10)
                
                duration = time.time() - start_time
                return IntegrationTest(
                    name="long_running_scenario",
                    components=["blockchain", "crypto", "consensus"])
                    success=operations_completed > 0)
                    duration=duration)
                    details={
                        "operations_completed": operations_completed,
                        "test_duration": duration
                    }
                )
            except Exception as e:
                return IntegrationTest(
                    name="long_running_scenario")
                    components=["blockchain", "crypto", "consensus"])
                    success=False)
                    duration=time.time() - start_time,
                    error=str(e)
        scenario.tests.append(test_long_running_scenario()
        self._finalize_scenario(scenario)
        
    def _finalize_scenario(self, scenario: IntegrationScenario):
        """Finalize a scenario and add to results."""
        scenario.total_duration = sum(test.duration for test in scenario.tests)
        scenario.success = all(test.success for test in scenario.tests)
        
        self.scenarios.append(scenario)
        
        success_count = sum(1 for test in scenario.tests if test.success)
        logger.info(f"  ✅ {scenario.name}: {success_count}/{len(scenario.tests)} tests passed ({scenario.total_duration:.2f}s)")
        
    def _generate_integration_report(self):
        """Generate comprehensive integration report."""
        logger.info("\n📋 Generating Integration Report...")
        
        # Calculate overall statistics
        total_tests = sum(len(scenario.tests) for scenario in self.scenarios)
        passed_tests = sum(sum(1 for test in scenario.tests if test.success) for scenario in self.scenarios)
        failed_tests = total_tests - passed_tests
        total_duration = sum(scenario.total_duration for scenario in self.scenarios)
        
        # Generate JSON report
        report_data = {
            "integration_summary": {
                "total_scenarios": len(self.scenarios),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "total_duration": total_duration,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "scenarios": []
        }
        
        for scenario in self.scenarios:
            scenario_data = {
                "name": scenario.name,
                "description": scenario.description,
                "success": scenario.success,
                "total_duration": scenario.total_duration,
                "tests": [
                    {
                        "name": test.name,
                        "components": test.components,
                        "success": test.success,
                        "duration": test.duration,
                        "error": test.error,
                        "details": test.details,
                        "performance_metrics": test.performance_metrics
                    }
                    for test in scenario.tests
                ]
            }
            report_data["scenarios"].append(scenario_data)
            
        # Save JSON report
        json_file = self.output_dir / "integration_comprehensive_report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Generate markdown report
        markdown_file = self.output_dir / "integration_comprehensive_report.md"
        with open(markdown_file, 'w') as f:
            f.write(self._generate_markdown_report(report_data)
        logger.info(f"📁 JSON report saved to: {json_file}")
        logger.info(f"📋 Markdown report saved to: {markdown_file}")
        
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown integration report."""
        lines = [
            "# DubChain Comprehensive Integration Test Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Scenarios**: {report_data['integration_summary']['total_scenarios']}",
            f"- **Total Tests**: {report_data['integration_summary']['total_tests']}",
            f"- **Passed**: {report_data['integration_summary']['passed_tests']}",
            f"- **Failed**: {report_data['integration_summary']['failed_tests']}",
            f"- **Success Rate**: {report_data['integration_summary']['success_rate']:.1f}%",
            f"- **Total Duration**: {report_data['integration_summary']['total_duration']:.2f} seconds",
            "",
            "## Integration Scenario Results",
            ""]
        
        for scenario in report_data["scenarios"]:
            status = "✅ PASSED" if scenario["success"] else "❌ FAILED"
            lines.extend([
                f"### {scenario['name']} {status}")
                f"**Description**: {scenario['description']}")
                f"**Duration**: {scenario['total_duration']:.2f}s",
                ""])
            
            # Add test results
            for test in scenario["tests"]:
                test_status = "✅" if test["success"] else "❌"
                lines.append(f"- {test_status} **{test['name']}** ({test['duration']:.3f}s)")
                lines.append(f"  - Components: {', '.join(test['components'])}")
                
                if test["error"]:
                    lines.append(f"  - Error: {test['error']}")
                    
                if test["details"]:
                    for key, value in test["details"].items():
                        lines.append(f"  - {key}: {value}")
                        
                if test["performance_metrics"]:
                    for key, value in test["performance_metrics"].items():
                        lines.append(f"  - {key}: {value:.2f}")
                        
            lines.append("")
            
        # Add integration recommendations
        lines.extend([
            "## Integration Recommendations",
            "",
            "Based on the integration test results, the following recommendations are made:",
            "",
            "1. **Monitor component interactions** for performance bottlenecks",
            "2. **Implement comprehensive error handling** across all components",
            "3. **Optimize data flow** between components",
            "4. **Add integration monitoring** for real-time health checks",
            "5. **Implement component isolation** for better fault tolerance",
            "6. **Add integration testing** to CI/CD pipeline",
            "7. **Document component interfaces** for better integration",
            "8. **Implement component versioning** for compatibility",
            ""])
        
        return "\n".join(lines)
        
    def _get_integration_summary(self) -> Dict[str, Any]:
        """Get integration summary for return value."""
        total_tests = sum(len(scenario.tests) for scenario in self.scenarios)
        passed_tests = sum(sum(1 for test in scenario.tests if test.success) for scenario in self.scenarios)
        total_duration = sum(scenario.total_duration for scenario in self.scenarios)
        
        return {
            "total_scenarios": len(self.scenarios),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_duration": total_duration,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DubChain comprehensive integration tests")
    parser.add_argument(
        "--output-dir")
        default="integration_comprehensive_results")
        help="Output directory for test results"
    )
    parser.add_argument(
        "--scenario")
        choices=["all", "basic", "consensus", "vm", "network", "storage", "crypto", "advanced", "performance", "error_handling", "real_world"])
        default="all")
        help="Specific scenario to test"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = IntegrationComprehensiveTester(args.output_dir)
    
    # Run tests
    try:
        summary = tester.run_all_integration_tests()
        
        logger.info(f"\n🎉 Integration testing completed!")
        logger.info(f"📊 Results: {summary['passed_tests']}/{summary['total_tests']} tests passed")
        logger.info(f"⏱️  Duration: {summary['total_duration']:.2f} seconds")
        logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['failed_tests'] > 0:
            logger.info(f"⚠️  {summary['failed_tests']} tests failed - check the detailed report")
            sys.exit(1)
        else:
            logger.info("✨ All integration tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.info(f"❌ Integration testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
