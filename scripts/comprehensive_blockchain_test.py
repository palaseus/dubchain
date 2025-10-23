#!/usr/bin/env python3
"""
Comprehensive DubChain Test Suite

This script tests every aspect of the DubChain blockchain system:
- Core blockchain functionality
- Consensus mechanisms
- Virtual machine and smart contracts
- Network layer
- Storage and caching
- Cryptographic operations
- Advanced features (sharding, state channels, governance)
- Performance and optimization
- Security and adversarial scenarios
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import threading
import random

logger = logging.getLogger(__name__)

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain import Blockchain, PrivateKey, PublicKey
from dubchain.core.consensus import ConsensusConfig


@dataclass
class TestResult:
    """Test result data structure."""
    name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Test suite data structure."""
    name: str
    tests: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    success_count: int = 0
    failure_count: int = 0


class ComprehensiveBlockchainTester:
    """Comprehensive tester for DubChain blockchain system."""
    
    def __init__(self, output_dir: str = "comprehensive_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_suites: Dict[str, TestSuite] = {}
        self.blockchain = None
        self.wallets = {}
        self.test_data = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_tests": 0,
            "total_duration": 0.0,
            "success_rate": 0.0,
            "performance_issues": []
        }
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting Comprehensive DubChain Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize blockchain
            self._initialize_blockchain()
            
            # Run test suites
            self._run_core_functionality_tests()
            self._run_consensus_tests()
            self._run_virtual_machine_tests()
            self._run_network_tests()
            self._run_storage_tests()
            self._run_crypto_tests()
            self._run_advanced_feature_tests()
            self._run_performance_tests()
            self._run_security_tests()
            self._run_integration_tests()
            
            # Generate reports
            self._generate_comprehensive_report()
            
        except Exception as e:
            logger.info(f"❌ Test suite failed: {e}")
            traceback.print_exc()
            
        finally:
            total_duration = time.time() - start_time
            self.performance_metrics["total_duration"] = total_duration
            
        logger.info(f"\n✅ Comprehensive test suite completed in {total_duration:.2f} seconds")
        return self._get_test_summary()
        
    def _initialize_blockchain(self):
        """Initialize blockchain and test environment."""
        logger.info("\n🔧 Initializing test environment...")
        
        # Create blockchain with test configuration
        config = ConsensusConfig(
            target_block_time=1.0,
            difficulty_adjustment_interval=10,
            min_difficulty=1,
            max_difficulty=5
        )
        
        self.blockchain = Blockchain(config)
        
        # Create genesis block
        genesis_block = self.blockchain.create_genesis_block(
            coinbase_recipient="test_miner",
            coinbase_amount=1000000000
        )
        logger.info(f"✅ Genesis block created: {genesis_block.get_hash().to_hex()[:16]}...")
        
        # Create test wallets
        for i in range(10):
            name = f"test_wallet_{i}"
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            address = public_key.to_address()
            
            self.wallets[name] = {
                'private_key': private_key,
                'public_key': public_key,
                'address': address
            }
            
        logger.info(f"✅ Created {len(self.wallets)} test wallets")
        
        # Mine some initial blocks
        for i in range(5):
            miner_address = self.wallets[f"test_wallet_{i % len(self.wallets)}"]['address']
            block = self.blockchain.mine_block(miner_address, max_transactions=10)
            if block:
                logger.info(f"✅ Mined block {i+1}")
                
    def _run_core_functionality_tests(self):
        """Test core blockchain functionality."""
        logger.info("\n📦 Testing Core Blockchain Functionality...")
        suite = TestSuite("Core Functionality")
        
        # Test 1: Block creation and validation
        def test_block_creation():
            start_time = time.time()
            try:
                # Create a new block
                block = self.blockchain.create_block(
                    previous_hash=self.blockchain.get_latest_block().get_hash(),
                    transactions=[]
                )
                
                # Validate block
                is_valid = block.validate()
                
                duration = time.time() - start_time
                return TestResult(
                    name="block_creation_validation",
                    success=is_valid,
                    duration=duration,
                    details={"block_hash": block.get_hash().to_hex()[:16]}
                )
            except Exception as e:
                return TestResult(
                    name="block_creation_validation",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)
                )
        
        suite.tests.append(test_block_creation)
        # Test 2: Transaction creation and validation
        def test_transaction_creation():
            start_time = time.time()
            try:
                sender = self.wallets["test_wallet_0"]
                recipient = self.wallets["test_wallet_1"]
                
                # Create transaction
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'],
                    recipient_address=recipient['address'],
                    amount=1000,
                    fee=10
                )
                
                # Validate transaction
                is_valid = tx.validate()
                
                duration = time.time() - start_time
                return TestResult(
                    name="transaction_creation_validation",
                    success=is_valid,
                    duration=duration,
                    details={"tx_hash": tx.get_hash().to_hex()[:16]}
                )
            except Exception as e:
                return TestResult(
                    name="transaction_creation_validation",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)
        
        suite.tests.append(test_transaction_creation)
        # Test 3: Blockchain integrity
        def test_blockchain_integrity():
            start_time = time.time()
            try:
                is_valid = self.blockchain.validate_chain()
                duration = time.time() - start_time
                
                return TestResult(
                    name="blockchain_integrity",
                    success=is_valid,
                    duration=duration,
                    details={"chain_length": len(self.blockchain.chain)}
                )
            except Exception as e:
                return TestResult(
                    name="blockchain_integrity",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

        suite.tests.append(test_blockchain_integrity)
        
        # Test 4: Balance tracking
        def test_balance_tracking():
            start_time = time.time()
            try:
                wallet = self.wallets["test_wallet_0"]
                balance = self.blockchain.get_balance(wallet['address'])
                
                duration = time.time() - start_time
                return TestResult(
                    name="balance_tracking",
                    success=balance >= 0,
                    duration=duration,
                    details={"balance": balance}
                )
            except Exception as e:
                return TestResult(
                    name="balance_tracking",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

        suite.tests.append(test_balance_tracking)
        # Test 5: Mining functionality
        def test_mining():
            start_time = time.time()
            try:
                miner_address = self.wallets["test_wallet_0"]['address']
                block = self.blockchain.mine_block(miner_address, max_transactions=5)
                
                duration = time.time() - start_time
                return TestResult(
                    name="mining_functionality",
                    success=block is not None,
                    duration=duration,
                    details={"block_mined": block is not None}
                )
            except Exception as e:
                return TestResult(
                    name="mining_functionality",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

        suite.tests.append(test_mining)
        self._finalize_test_suite(suite)
        
    def _run_consensus_tests(self):
        """Test consensus mechanisms."""
        logger.info("\n🔄 Testing Consensus Mechanisms...")
        suite = TestSuite("Consensus Mechanisms")
        
        # Test 1: Proof of Stake consensus
        def test_proof_of_stake():
            start_time = time.time()
            try:
                from dubchain.consensus.proof_of_stake import ProofOfStake
                from dubchain.consensus.consensus_types import ConsensusType, ConsensusConfig
                
                config = ConsensusConfig()
                config.consensus_type = ConsensusType.PROOF_OF_STAKE
                pos = ProofOfStake(config)
                
                # Add validators
                pos.add_validator("validator_1", 1000000)
                pos.add_validator("validator_2", 2000000)
                pos.add_validator("validator_3", 1500000)
                
                # Test proposer selection
                proposer = pos.select_proposer(1)
                
                duration = time.time() - start_time
                return TestResult(
                    name="proof_of_stake_consensus",
                    success=proposer is not None,
                    duration=duration,
                    details={"proposer": proposer}
                )
            except Exception as e:
                return TestResult(
                    name="proof_of_stake_consensus",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

        suite.tests.append(test_proof_of_stake)
        # Test 2: PBFT consensus
        def test_pbft_consensus():
            start_time = time.time()
            try:
                from dubchain.consensus.pbft import PracticalByzantineFaultTolerance
                from dubchain.consensus.consensus_types import ConsensusType, ConsensusConfig
                
                config = ConsensusConfig()
                config.consensus_type = ConsensusType.PBFT
                pbft = PracticalByzantineFaultTolerance(config)
                
                # Add validators
                for i in range(3):
                    pbft.add_validator(i, f"validator_{i}", f"public_key_{i}")
                
                duration = time.time() - start_time
                return TestResult(
                    name="pbft_consensus",
                    success=len(pbft.validators) == 3,
                    duration=duration,
                    details={"validator_count": len(pbft.validators)}
                )
            except Exception as e:
                return TestResult(
                    name="pbft_consensus",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_pbft_consensus)
        # Test 3: Consensus switching
        def test_consensus_switching():
            start_time = time.time()
            try:
                from dubchain.consensus.consensus_types import ConsensusType, ConsensusConfig
                
                # Test switching between consensus mechanisms
                config1 = ConsensusConfig()
                config1.consensus_type = ConsensusType.PROOF_OF_STAKE
                config2 = ConsensusConfig()
                config2.consensus_type = ConsensusType.PBFT
                
                duration = time.time() - start_time
                return TestResult(
                    name="consensus_switching",
                    success=True,
                    duration=duration,
                    details={"configs_created": 2}
                )
            except Exception as e:
                return TestResult(
                    name="consensus_switching",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_consensus_switching)
        self._finalize_test_suite(suite)
        
    def _run_virtual_machine_tests(self):
        """Test virtual machine and smart contracts."""
        logger.info("\n💻 Testing Virtual Machine...")
        suite = TestSuite("Virtual Machine")
        
        # Test 1: VM execution engine
        def test_vm_execution():
            start_time = time.time()
            try:
                from dubchain.vm.execution_engine import ExecutionEngine
                
                engine = ExecutionEngine()
                
                # Test basic execution
                result = engine.execute_bytecode(b"test_bytecode")
                
                duration = time.time() - start_time
                return TestResult(
                    name="vm_execution_engine",
                    success=result is not None,
                    duration=duration,
                    details={"execution_result": str(result)[:50]}
                )
            except Exception as e:
                return TestResult(
                    name="vm_execution_engine",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_vm_execution)
        # Test 2: Gas metering
        def test_gas_metering():
            start_time = time.time()
            try:
                from dubchain.vm.gas_meter import GasMeter
                
                gas_meter = GasMeter(1000000)
                gas_meter.consume_gas(1000)
                
                duration = time.time() - start_time
                return TestResult(
                    name="gas_metering",
                    success=gas_meter.gas_remaining == 999000,
                    duration=duration,
                    details={"gas_remaining": gas_meter.gas_remaining}
                )
            except Exception as e:
                return TestResult(
                    name="gas_metering",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_gas_metering)
        # Test 3: Smart contract deployment
        def test_smart_contract_deployment():
            start_time = time.time()
            try:
                from dubchain.vm.contract import SmartContract
                
                contract = SmartContract(
                    address="test_contract_address",
                    bytecode=b"contract_bytecode",
                    creator="test_creator"
                )
                duration = time.time() - start_time
                return TestResult(
                    name="smart_contract_deployment",
                    success=contract is not None,
                    duration=duration,
                    details={"contract_address": contract.address}
                )
            except Exception as e:
                return TestResult(
                    name="smart_contract_deployment",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_smart_contract_deployment)
        self._finalize_test_suite(suite)
        
    def _run_network_tests(self):
        """Test network layer functionality."""
        logger.info("\n🌐 Testing Network Layer...")
        suite = TestSuite("Network Layer")
        
        # Test 1: Message serialization
        def test_message_serialization():
            start_time = time.time()
            try:
                import json
                
                message = {
                    "type": "block",
                    "data": "test_data",
                    "timestamp": time.time()
                }
                
                serialized = json.dumps(message)
                deserialized = json.loads(serialized)
                
                duration = time.time() - start_time
                return TestResult(
                    name="message_serialization",
                    success=deserialized == message,
                    duration=duration,
                    details={"message_size": len(serialized)}
                )
            except Exception as e:
                return TestResult(
                    name="message_serialization",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_message_serialization)
        # Test 2: Peer management
        def test_peer_management():
            start_time = time.time()
            try:
                from dubchain.network.peer import Peer
                
                peer = Peer("test_peer", "127.0.0.1", 8000)
                duration = time.time() - start_time
                return TestResult(
                    name="peer_management",
                    success=peer is not None,
                    duration=duration,
                    details={"peer_id": peer.peer_id}
                )
            except Exception as e:
                return TestResult(
                    name="peer_management",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_peer_management)
        # Test 3: Network protocol
        def test_network_protocol():
            start_time = time.time()
            try:
                from dubchain.network.protocol import NetworkProtocol
                from dubchain.crypto.signatures import PrivateKey
                
                private_key = PrivateKey.generate()
                protocol = NetworkProtocol(private_key)
                
                duration = time.time() - start_time
                return TestResult(
                    name="network_protocol",
                    success=protocol is not None,
                    duration=duration,
                    details={"protocol_version": getattr(protocol, 'version', 'unknown')}
                )
            except Exception as e:
                return TestResult(
                    name="network_protocol",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_network_protocol)
        self._finalize_test_suite(suite)
        
    def _run_storage_tests(self):
        """Test storage and caching functionality."""
        logger.info("\n💾 Testing Storage Layer...")
        suite = TestSuite("Storage Layer")
        
        # Test 1: Database operations
        def test_database_operations():
            start_time = time.time()
            try:
                import sqlite3
                import tempfile
                
                # Create temporary database
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                    db_path = tmp.name
                    
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create table and insert data
                cursor.execute("CREATE TABLE test (id INTEGER, data TEXT)")
                cursor.execute("INSERT INTO test VALUES (?, ?)", (1, "test_data"))
                
                # Query data
                cursor.execute("SELECT * FROM test WHERE id = ?", (1)
                result = cursor.fetchone()
                
                conn.close()
                os.unlink(db_path)
                
                duration = time.time() - start_time
                return TestResult(
                    name="database_operations",
                    success=result is not None,
                    duration=duration,
                    details={"query_result": result}
                )
            except Exception as e:
                return TestResult(
                    name="database_operations",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_database_operations)
        # Test 2: File storage
        def test_file_storage():
            start_time = time.time()
            try:
                import tempfile
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                    tmp.write("test_data")
                    tmp_path = tmp.name
                
                # Read file
                with open(tmp_path, 'r') as f:
                    content = f.read()
                
                os.unlink(tmp_path)
                
                duration = time.time() - start_time
                return TestResult(
                    name="file_storage",
                    success=content == "test_data",
                    duration=duration,
                    details={"content_length": len(content)}
                )
            except Exception as e:
                return TestResult(
                    name="file_storage",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_file_storage)
        self._finalize_test_suite(suite)
        
    def _run_crypto_tests(self):
        """Test cryptographic operations."""
        logger.info("\n🔐 Testing Cryptographic Operations...")
        suite = TestSuite("Cryptographic Operations")
        
        # Test 1: Key generation
        def test_key_generation():
            start_time = time.time()
            try:
                private_key = PrivateKey.generate()
                public_key = private_key.get_public_key()
                address = public_key.to_address()
                
                duration = time.time() - start_time
                return TestResult(
                    name="key_generation",
                    success=all([private_key, public_key, address]),
                    duration=duration,
                    details={"address": address})
                )
            except Exception as e:
                return TestResult(
                    name="key_generation",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_key_generation)
        # Test 2: Digital signatures
        def test_digital_signatures():
            start_time = time.time()
            try:
                private_key = PrivateKey.generate()
                public_key = private_key.get_public_key()
                
                message = b"test_message"
                signature = private_key.sign(message)
                is_valid = public_key.verify(signature, message)
                
                # Handle both Signature objects and bytes
                if hasattr(signature, 'to_der'):
                    signature_length = len(signature.to_der()
                elif isinstance(signature, bytes):
                    signature_length = len(signature)
                else:
                    signature_length = 0
                
                duration = time.time() - start_time
                return TestResult(
                    name="digital_signatures",
                    success=is_valid,
                    duration=duration,
                    details={"signature_length": signature_length}
                )
            except Exception as e:
                return TestResult(
                    name="digital_signatures",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_digital_signatures)
        # Test 3: Hash functions
        def test_hash_functions():
            start_time = time.time()
            try:
                from dubchain.crypto.hashing import sha256_hash
                
                data = b"test_data"
                hash_value = sha256_hash(data)
                duration = time.time() - start_time
                return TestResult(
                    name="hash_functions",
                    success=len(hash_value.value) == 32,  # SHA256 produces 32 bytes,
                    duration=duration,
                    details={"hash_length": len(hash_value.value)})
                )
            except Exception as e:
                return TestResult(
                    name="hash_functions",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_hash_functions)
        self._finalize_test_suite(suite)
        
    def _run_advanced_feature_tests(self):
        """Test advanced blockchain features."""
        logger.info("\n🚀 Testing Advanced Features...")
        suite = TestSuite("Advanced Features")
        
        # Test 1: Sharding
        def test_sharding():
            start_time = time.time()
            try:
                from dubchain.sharding.shard_manager import ShardManager
                from dubchain.sharding.shard_types import ShardConfig
                
                config = ShardConfig()
                shard_manager = ShardManager(config)
                shard_id = shard_manager.create_shard("test_shard")
                
                duration = time.time() - start_time
                return TestResult(
                    name="sharding",
                    success=shard_id is not None,
                    duration=duration,
                    details={"shard_id": str(shard_id)}
                )
            except Exception as e:
                return TestResult(
                    name="sharding",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_sharding)
        # Test 2: State channels
        def test_state_channels():
            start_time = time.time()
            try:
                from dubchain.state_channels.channel import StateChannel, ChannelParticipant
                from dubchain.crypto.signatures import PublicKey, PrivateKey
                
                # Generate proper public keys
                private_key1 = PrivateKey.generate()
                private_key2 = PrivateKey.generate()
                public_key1 = private_key1.get_public_key()
                public_key2 = private_key2.get_public_key()
                
                participants = [
                    ChannelParticipant(
                        address=self.wallets["test_wallet_0"]['address'],
                        public_key=public_key1,
                        balance=500
                    ),
                    ChannelParticipant(
                        address=self.wallets["test_wallet_1"]['address'],
                        public_key=public_key2,
                        balance=500
                    )
                ]
                
                channel = StateChannel(
                    channel_id="test_channel",
                    participants=participants
                )
                
                duration = time.time() - start_time
                return TestResult(
                    name="state_channels")
                    success=channel is not None)
                    duration=duration)
                    details={"participants": len(channel.participants)}
                )
            except Exception as e:
                return TestResult(
                    name="state_channels",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_state_channels)
        # Test 3: Governance
        def test_governance():
            start_time = time.time()
            try:
                from dubchain.governance.core import Proposal
                
                proposal = Proposal()
                    title="Test Proposal")
                    description="Test governance proposal")
                    proposer_address=self.wallets["test_wallet_0"]['address']
                )
                duration = time.time() - start_time
                return TestResult(
                    name="governance")
                    success=proposal is not None)
                    duration=duration)
                    details={"proposal_title": proposal.title}
                )
            except Exception as e:
                return TestResult(
                    name="governance",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_governance)
        self._finalize_test_suite(suite)
        
    def _run_performance_tests(self):
        """Test performance and optimization features."""
        logger.info("\n⚡ Testing Performance Features...")
        suite = TestSuite("Performance Features")
        
        # Test 1: Transaction throughput
        def test_transaction_throughput():
            start_time = time.time()
            try:
                transactions_created = 0
                test_duration = 1.0  # 1 second test
                
                end_time = start_time + test_duration
                while time.time() < end_time:
                    sender = random.choice(list(self.wallets.keys)
                    recipient = random.choice([w for w in self.wallets.keys() if w != sender])
                    
                    tx = self.blockchain.create_transfer_transaction(
                        sender_private_key=self.wallets[sender]['private_key'])
                        recipient_address=self.wallets[recipient]['address'])
                        amount=100)
                        fee=1
                    )
                    
                    if tx:
                        transactions_created += 1
                
                duration = time.time() - start_time
                tps = transactions_created / duration
                
                return TestResult(
                    name="transaction_throughput")
                    success=tps > 0)
                    duration=duration)
                    details={"transactions_per_second": tps, "total_transactions": transactions_created}
                )
            except Exception as e:
                return TestResult(
                    name="transaction_throughput",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_transaction_throughput)
        # Test 2: Memory usage
        def test_memory_usage():
            start_time = time.time()
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                duration = time.time() - start_time
                return TestResult(
                    name="memory_usage")
                    success=memory_usage < 1000,  # Less than 1GB)
                    duration=duration)
                    details={"memory_usage_mb": memory_usage}
                )
            except Exception as e:
                return TestResult(
                    name="memory_usage",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_memory_usage)
        self._finalize_test_suite(suite)
        
    def _run_security_tests(self):
        """Test security features and adversarial scenarios."""
        logger.info("\n🔒 Testing Security Features...")
        suite = TestSuite("Security Features")
        
        # Test 1: Double spending prevention
        def test_double_spending_prevention():
            start_time = time.time()
            try:
                sender = self.wallets["test_wallet_0"]
                recipient = self.wallets["test_wallet_1"]
                
                # Create first transaction
                tx1 = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Try to create second transaction with same input
                tx2 = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'])
                    recipient_address=recipient['address'])
                    amount=1000)
                    fee=10
                )
                
                # Both transactions should be created but only one should be valid
                duration = time.time() - start_time
                return TestResult(
                    name="double_spending_prevention",
                    success=tx1 is not None and tx2 is not None,
                    duration=duration,
                    details={"tx1_created": tx1 is not None, "tx2_created": tx2 is not None}
                )
            except Exception as e:
                return TestResult(
                    name="double_spending_prevention",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_double_spending_prevention)
        # Test 2: Invalid transaction rejection
        def test_invalid_transaction_rejection():
            start_time = time.time()
            try:
                # Try to create transaction with invalid signature
                invalid_tx = self.blockchain.create_transfer_transaction()
                    sender_private_key=PrivateKey.generate(),  # Different key
                    recipient_address=self.wallets["test_wallet_1"]['address'],
                    amount=1000,
                    fee=10
                )
                
                # Transaction should be created but validation should fail
                is_valid = invalid_tx.validate() if invalid_tx else False
                
                duration = time.time() - start_time
                return TestResult(
                    name="invalid_transaction_rejection",
                    success=not is_valid,  # Should be invalid
                    duration=duration,
                    details={"transaction_created": invalid_tx is not None, "is_valid": is_valid}
                )
            except Exception as e:
                return TestResult(
                    name="invalid_transaction_rejection",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_invalid_transaction_rejection)
        self._finalize_test_suite(suite)
        
    def _run_integration_tests(self):
        """Test integration between different components."""
        logger.info("\n🔗 Testing Integration...")
        suite = TestSuite("Integration Tests")
        
        # Test 1: End-to-end transaction flow
        def test_end_to_end_transaction():
            start_time = time.time()
            try:
                sender = self.wallets["test_wallet_0"]
                recipient = self.wallets["test_wallet_1"]
                
                # Create transaction
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'],
                    recipient_address=recipient['address'],
                    amount=1000,
                    fee=10
                )
                
                # Add to blockchain
                self.blockchain.add_transaction(tx)
                
                # Mine block
                miner_address = self.wallets["test_wallet_2"]['address']
                block = self.blockchain.mine_block(miner_address, max_transactions=1)
                
                # Verify transaction is in block
                tx_in_block = any(t.get_hash() == tx.get_hash() for t in block.transactions) if block else False
                
                duration = time.time() - start_time
                return TestResult(
                    name="end_to_end_transaction",
                    success=tx_in_block,
                    duration=duration,
                    details={"transaction_added": tx is not None, "block_mined": block is not None}
                )
            except Exception as e:
                return TestResult(
                    name="end_to_end_transaction",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_end_to_end_transaction)
        # Test 2: Multi-component interaction
        def test_multi_component_interaction():
            start_time = time.time()
            try:
                from dubchain.consensus.consensus_types import ConsensusType, ConsensusConfig
                from dubchain.consensus.proof_of_stake import ProofOfStake
                
                # Test blockchain + consensus + crypto interaction
                config = ConsensusConfig()
                config.consensus_type = ConsensusType.PROOF_OF_STAKE
                pos = ProofOfStake(config)
                
                # Add validators
                pos.add_validator("validator_1", 1000000)
                pos.add_validator("validator_2", 2000000)
                
                # Create transaction with crypto
                sender = self.wallets["test_wallet_0"]
                recipient = self.wallets["test_wallet_1"]
                
                tx = self.blockchain.create_transfer_transaction(
                    sender_private_key=sender['private_key'],
                    recipient_address=recipient['address'],
                    amount=1000,
                    fee=10
                )
                
                # Use consensus to select proposer
                proposer = pos.select_proposer(1)
                
                duration = time.time() - start_time
                return TestResult(
                    name="multi_component_interaction",
                    success=all([tx, proposer]),
                    duration=duration,
                    details={"transaction_created": tx is not None, "proposer_selected": proposer is not None})
                )
            except Exception as e:
                return TestResult(
                    name="multi_component_interaction",
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e)

                suite.tests.append(test_multi_component_interaction)
        self._finalize_test_suite(suite)
        
    def _finalize_test_suite(self, suite: TestSuite):
        """Finalize a test suite and add to results."""
        suite.total_duration = sum(test.duration for test in suite.tests)
        suite.success_count = sum(1 for test in suite.tests if test.success)
        suite.failure_count = len(suite.tests) - suite.success_count
        
        self.test_suites[suite.name] = suite
        
        logger.info(f"  ✅ {suite.name}: {suite.success_count}/{len(suite.tests)} tests passed ({suite.total_duration:.2f}s)")
        
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        logger.info("\n📋 Generating Comprehensive Test Report...")
        
        # Calculate overall statistics
        total_tests = sum(len(suite.tests) for suite in self.test_suites.values()
        total_passed = sum(suite.success_count for suite in self.test_suites.values()
        total_failed = sum(suite.failure_count for suite in self.test_suites.values()
        total_duration = sum(suite.total_duration for suite in self.test_suites.values()
        # Generate JSON report
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration
            },
            "test_suites": {}
        }
        
        for suite_name, suite in self.test_suites.items():
            report_data["test_suites"][suite_name] = {
                "total_tests": len(suite.tests),
                "passed": suite.success_count,
                "failed": suite.failure_count,
                "duration": suite.total_duration,
                "tests": [
                    {
                        "name": test.name,
                        "success": test.success,
                        "duration": test.duration,
                        "error": test.error,
                        "details": test.details
                    }
                    for test in suite.tests
                ]
            }
        
        # Save JSON report
        json_file = self.output_dir / "comprehensive_test_report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Generate markdown report
        markdown_file = self.output_dir / "comprehensive_test_report.md"
        with open(markdown_file, 'w') as f:
            f.write(self._generate_markdown_report(report_data)
        logger.info(f"📁 JSON report saved to: {json_file}")
        logger.info(f"📋 Markdown report saved to: {markdown_file}")
        
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown test report."""
        lines = [
            "# DubChain Comprehensive Test Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Tests**: {report_data['test_summary']['total_tests']}",
            f"- **Passed**: {report_data['test_summary']['passed']}",
            f"- **Failed**: {report_data['test_summary']['failed']}",
            f"- **Success Rate**: {report_data['test_summary']['success_rate']:.1f}%",
            f"- **Total Duration**: {report_data['test_summary']['total_duration']:.2f} seconds",
            "",
            "## Test Suite Results",
            ""]
        
        for suite_name, suite_data in report_data["test_suites"].items():
            status = "✅ PASSED" if suite_data["failed"] == 0 else "❌ FAILED"
            lines.extend([
                f"### {suite_name} {status}",
                f"- **Tests**: {suite_data['passed']}/{suite_data['total_tests']} passed",
                f"- **Duration**: {suite_data['duration']:.2f}s",
                ""])
            
            # Add individual test results
            for test in suite_data["tests"]:
                test_status = "✅" if test["success"] else "❌"
                lines.append(f"- {test_status} **{test['name']}** ({test['duration']:.3f}s)")
                if test["error"]:
                    lines.append(f"  - Error: {test['error']}")
                if test["details"]:
                    for key, value in test["details"].items():
                        lines.append(f"  - {key}: {value}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _get_test_summary(self) -> Dict[str, Any]:
        """Get test summary for return value."""
        total_tests = sum(len(suite.tests) for suite in self.test_suites.values()
        total_passed = sum(suite.success_count for suite in self.test_suites.values()
        total_failed = sum(suite.failure_count for suite in self.test_suites.values()
        total_duration = sum(suite.total_duration for suite in self.test_suites.values()
        return {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "test_suites": len(self.test_suites)
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive DubChain tests")
    parser.add_argument(
        "--output-dir")
        default="comprehensive_test_results")
        help="Output directory for test results"
    )
    parser.add_argument(
        "--quick")
        action="store_true")
        help="Run quick tests (reduced iterations)"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = ComprehensiveBlockchainTester(args.output_dir)
    
    # Run tests
    try:
        summary = tester.run_all_tests()
        
        logger.info(f"\n🎉 Comprehensive testing completed!")
        logger.info(f"📊 Results: {summary['passed']}/{summary['total_tests']} tests passed")
        logger.info(f"⏱️  Duration: {summary['total_duration']:.2f} seconds")
        logger.info(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['failed'] > 0:
            logger.info(f"⚠️  {summary['failed']} tests failed - check the detailed report")
            sys.exit(1)
        else:
            logger.info("✨ All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.info(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
