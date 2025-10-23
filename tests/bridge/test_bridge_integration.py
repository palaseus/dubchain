"""
Comprehensive Integration Tests for Blockchain Bridges

This module provides comprehensive integration tests for all blockchain bridge components including:
- Ethereum bridge integration tests
- Bitcoin bridge integration tests
- Polygon bridge integration tests
- BSC bridge integration tests
- Universal bridge integration tests
- Bridge validator network tests
- Cross-chain transaction tests
- Bridge security and fraud detection tests
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import unittest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional
import tempfile
import os
import sys
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.dubchain.bridge.chains.ethereum import EthereumClient, EthereumConfig, EthereumTransaction
from src.dubchain.bridge.chains.bitcoin import BitcoinClient, BitcoinConfig, BitcoinTransaction
from src.dubchain.bridge.chains.polygon import PolygonClient, PolygonConfig, PolygonTransaction
from src.dubchain.bridge.chains.bsc import BSCClient, BSCConfig, BSCTransaction
from src.dubchain.bridge.universal import UniversalBridge, UniversalTransaction, ChainType, TokenType
from src.dubchain.bridge.validators import BridgeValidatorNetwork, ValidatorConfig, Validator
from src.dubchain.bridge.atomic_swap import AtomicSwapManager, SwapProposal, SwapStatus


class TestEthereumBridgeIntegration(unittest.TestCase):
    """Test Ethereum bridge integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = EthereumConfig(
            rpc_url="https://eth-mainnet.g.alchemy.com/v2/test",
            chain_id=1,
            enable_poa_middleware=False
        )
        self.client = EthereumClient(self.config)
    
    def test_client_initialization(self):
        """Test Ethereum client initialization."""
        self.assertIsInstance(self.client, EthereumClient)
        self.assertIsInstance(self.client.config, EthereumConfig)
    
    def test_connection_test(self):
        """Test Ethereum connection."""
        # Mock the is_connected method directly
        with patch.object(self.client, 'is_connected', return_value=True):
            connected = self.client.test_connection()
            self.assertTrue(connected)
    
    def test_gas_price_oracle(self):
        """Test gas price oracle functionality."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_web3.eth.gas_price = 20000000000  # 20 Gwei
            
            gas_price = self.client.get_gas_price()
            self.assertIsInstance(gas_price, int)
            self.assertGreater(gas_price, 0)
    
    def test_transaction_creation(self):
        """Test transaction creation."""
        tx = EthereumTransaction(
            hash="0x1234567890abcdef",
            from_address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            to_address="0x8ba1f109551bD432803012645Hac136c",
            value=1000000000000000000,  # 1 ETH
            gas=21000,
            gas_price=20000000000,
            nonce=1,
            data="0x"
        )
        
        self.assertIsInstance(tx, EthereumTransaction)
        self.assertEqual(tx.value, 1000000000000000000)
        self.assertEqual(tx.gas, 21000)
    
    def test_event_monitoring(self):
        """Test event monitoring functionality."""
        with patch.object(self.client, 'monitoring_service') as mock_monitoring:
            mock_monitoring.start = AsyncMock()
            mock_monitoring.stop = AsyncMock()
            mock_monitoring.get_monitoring_stats.return_value = {
                'is_running': True,
                'event_count': 100
            }
            
            # Test monitoring start
            asyncio.run(self.client.start_monitoring())
            mock_monitoring.start.assert_called_once()
            
            # Test monitoring stats
            stats = self.client.get_monitoring_stats()
            self.assertIn('is_running', stats)
            self.assertIn('event_count', stats)
    
    def test_erc20_token_operations(self):
        """Test ERC-20 token operations."""
        # Mock token contract interaction
        with patch.object(self.client, 'web3') as mock_web3:
            mock_contract = Mock()
            mock_contract.functions.balanceOf.return_value.call.return_value = 1000000
            mock_contract.functions.transfer.return_value.build_transaction.return_value = {
                'to': '0x1234567890123456789012345678901234567890',
                'data': '0x',
                'gas': 100000,
                'gasPrice': 20000000000
            }
            
            mock_web3.eth.contract.return_value = mock_contract
            
            # Test balance check
            balance = self.client.get_token_balance(
                "0x1234567890123456789012345678901234567890",
                "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
            )
            self.assertEqual(balance, 1000000)
    
    def test_erc721_nft_operations(self):
        """Test ERC-721 NFT operations."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_contract = Mock()
            mock_contract.functions.ownerOf.return_value.call.return_value = "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
            mock_contract.functions.safeTransferFrom.return_value.build_transaction.return_value = {
                'to': '0x1234567890123456789012345678901234567890',
                'data': '0x',
                'gas': 150000,
                'gasPrice': 20000000000
            }
            
            mock_web3.eth.contract.return_value = mock_contract
            
            # Test NFT owner check
            owner = self.client.get_nft_owner(
                "0x1234567890123456789012345678901234567890",
                1
            )
            self.assertEqual(owner, "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")


class TestBitcoinBridgeIntegration(unittest.TestCase):
    """Test Bitcoin bridge integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BitcoinConfig(
            rpc_host="localhost",
            rpc_port=18332,
            rpc_user="test",
            rpc_password="test",
            network="testnet"
        )
        self.client = BitcoinClient(self.config)
    
    def test_client_initialization(self):
        """Test Bitcoin client initialization."""
        self.assertIsInstance(self.client, BitcoinClient)
        self.assertIsInstance(self.client.config, BitcoinConfig)
    
    def test_connection_test(self):
        """Test Bitcoin connection."""
        with patch.object(self.client, 'rpc_client') as mock_rpc:
            mock_rpc.getblockchaininfo.return_value = {
                'chain': 'test',
                'blocks': 1000000,
                'verificationprogress': 1.0
            }
            
            connected = self.client.test_connection()
            self.assertTrue(connected)
    
    def test_utxo_management(self):
        """Test UTXO management."""
        with patch.object(self.client, 'rpc_client') as mock_rpc:
            mock_rpc.listunspent.return_value = [
                {
                    'txid': '1234567890abcdef',
                    'vout': 0,
                    'address': 'tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx',
                    'amount': 0.001,
                    'confirmations': 6,
                    'scriptPubKey': '0014751e76e8199196d454941c45d1b3a323f1433bd6'
                }
            ]
            
            utxos = self.client.get_utxos("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx")
            self.assertIsInstance(utxos, list)
            self.assertEqual(len(utxos), 1)
            self.assertEqual(utxos[0].amount_satoshi, 100000)
    
    def test_transaction_creation(self):
        """Test Bitcoin transaction creation."""
        tx = BitcoinTransaction(
            txid="1234567890abcdef",
            inputs=[{
                'txid': 'abcdef1234567890',
                'vout': 0,
                'amount': 100000
            }],
            outputs=[{
                'address': 'tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx',
                'amount': 95000
            }],
            fee=5000,
            confirmations=0
        )
        
        self.assertIsInstance(tx, BitcoinTransaction)
        self.assertEqual(tx.fee, 5000)
        self.assertEqual(len(tx.inputs), 1)
        self.assertEqual(len(tx.outputs), 1)
    
    def test_segwit_support(self):
        """Test SegWit support."""
        with patch.object(self.client, 'rpc_client') as mock_rpc:
            mock_rpc.getaddressinfo.return_value = {
                'address': 'tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx',
                'scriptPubKey': '0014751e76e8199196d454941c45d1b3a323f1433bd6',
                'iswitness': True,
                'witness_version': 0
            }
            
            address_info = self.client.get_address_info("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx")
            self.assertTrue(address_info['iswitness'])
            self.assertEqual(address_info['witness_version'], 0)
    
    def test_multisig_support(self):
        """Test multi-signature support."""
        with patch.object(self.client, 'rpc_client') as mock_rpc:
            mock_rpc.createmultisig.return_value = {
                'address': '2N1Ffz3wjL4GjL4GjL4GjL4GjL4GjL4GjL4G',
                'redeemScript': '522103abcdef...',
                'descriptor': 'sh(multi(2,03abcdef...,03fedcba...))'
            }
            
            multisig_info = self.client.create_multisig_address(
                ["03abcdef...", "03fedcba..."],
                2
            )
            self.assertIn('address', multisig_info)
            self.assertIn('redeemScript', multisig_info)


class TestPolygonBridgeIntegration(unittest.TestCase):
    """Test Polygon bridge integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PolygonConfig(
            rpc_url="https://polygon-rpc.com",
            chain_id=137
        )
        self.client = PolygonClient(self.config)
    
    def test_client_initialization(self):
        """Test Polygon client initialization."""
        self.assertIsInstance(self.client, PolygonClient)
        self.assertIsInstance(self.client.config, PolygonConfig)
    
    def test_connection_test(self):
        """Test Polygon connection."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_web3.is_connected.return_value = True
            mock_web3.eth.chain_id = 137
            
            connected = self.client.test_connection()
            self.assertTrue(connected)
    
    def test_pos_bridge_integration(self):
        """Test PoS bridge integration."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_contract = Mock()
            mock_contract.functions.depositFor.return_value.build_transaction.return_value = {
                'to': '0x8484Ef722627bf18ca5Ae6BcF031c23E6e922B30',
                'data': '0x',
                'gas': 200000,
                'gasPrice': 30000000000
            }
            
            mock_web3.eth.contract.return_value = mock_contract
            
            # Test deposit transaction
            tx_data = self.client.create_deposit_transaction(
                "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                1000000000000000000  # 1 MATIC
            )
            self.assertIn('to', tx_data)
            self.assertIn('data', tx_data)
    
    def test_zkevm_integration(self):
        """Test zkEVM integration."""
        with patch.object(self.client, 'zkevm_client') as mock_zkevm:
            mock_zkevm.submit_batch.return_value = {
                'batch_id': '0x1234567890abcdef',
                'status': 'submitted',
                'timestamp': int(time.time())
            }
            
            batch_result = self.client.submit_zkevm_batch({
                'transactions': ['0x1234', '0x5678'],
                'proof': '0xabcdef'
            })
            self.assertIn('batch_id', batch_result)
            self.assertEqual(batch_result['status'], 'submitted')


class TestBSCBridgeIntegration(unittest.TestCase):
    """Test BSC bridge integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BSCConfig(
            rpc_url="https://bsc-dataseed.binance.org",
            chain_id=56
        )
        self.client = BSCClient(self.config)
    
    def test_client_initialization(self):
        """Test BSC client initialization."""
        self.assertIsInstance(self.client, BSCClient)
        self.assertIsInstance(self.client.config, BSCConfig)
    
    def test_connection_test(self):
        """Test BSC connection."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_web3.is_connected.return_value = True
            mock_web3.eth.chain_id = 56
            
            connected = self.client.test_connection()
            self.assertTrue(connected)
    
    def test_bep20_token_operations(self):
        """Test BEP-20 token operations."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_contract = Mock()
            mock_contract.functions.balanceOf.return_value.call.return_value = 5000000
            mock_contract.functions.transfer.return_value.build_transaction.return_value = {
                'to': '0x1234567890123456789012345678901234567890',
                'data': '0x',
                'gas': 100000,
                'gasPrice': 5000000000  # 5 Gwei
            }
            
            mock_web3.eth.contract.return_value = mock_contract
            
            # Test balance check
            balance = self.client.get_token_balance(
                "0x1234567890123456789012345678901234567890",
                "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
            )
            self.assertEqual(balance, 5000000)
    
    def test_bep721_nft_operations(self):
        """Test BEP-721 NFT operations."""
        with patch.object(self.client, 'web3') as mock_web3:
            mock_contract = Mock()
            mock_contract.functions.ownerOf.return_value.call.return_value = "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
            mock_contract.functions.safeTransferFrom.return_value.build_transaction.return_value = {
                'to': '0x1234567890123456789012345678901234567890',
                'data': '0x',
                'gas': 150000,
                'gasPrice': 5000000000
            }
            
            mock_web3.eth.contract.return_value = mock_contract
            
            # Test NFT owner check
            owner = self.client.get_nft_owner(
                "0x1234567890123456789012345678901234567890",
                1
            )
            self.assertEqual(owner, "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")


class TestUniversalBridgeIntegration(unittest.TestCase):
    """Test universal bridge integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bridge = UniversalBridge()
        
        # Mock individual chain clients
        self.ethereum_client = Mock()
        self.bitcoin_client = Mock()
        self.polygon_client = Mock()
        self.bsc_client = Mock()
        
        # Register clients
        self.bridge.register_chain(ChainType.ETHEREUM, self.ethereum_client)
        self.bridge.register_chain(ChainType.BITCOIN, self.bitcoin_client)
        self.bridge.register_chain(ChainType.POLYGON, self.polygon_client)
        self.bridge.register_chain(ChainType.BSC, self.bsc_client)
    
    def test_bridge_initialization(self):
        """Test universal bridge initialization."""
        self.assertIsInstance(self.bridge, UniversalBridge)
    
    def test_chain_registration(self):
        """Test chain registration."""
        self.assertIn(ChainType.ETHEREUM, self.bridge.chains)
        self.assertIn(ChainType.BITCOIN, self.bridge.chains)
        self.assertIn(ChainType.POLYGON, self.bridge.chains)
        self.assertIn(ChainType.BSC, self.bridge.chains)
    
    def test_universal_transaction_creation(self):
        """Test universal transaction creation."""
        tx = UniversalTransaction(
            tx_id="0x1234567890abcdef",
            from_address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            to_address="0x8ba1f109551bD432803012645Hac136c",
            amount=1000000000000000000,
            token_type=TokenType.NATIVE,
            chain_type=ChainType.ETHEREUM,
            created_at=time.time()
        )
        
        self.assertIsInstance(tx, UniversalTransaction)
        self.assertEqual(tx.chain_type, ChainType.ETHEREUM)
        self.assertEqual(tx.token_type, TokenType.NATIVE)
    
    def test_cross_chain_transfer(self):
        """Test cross-chain transfer."""
        # Mock chain clients
        self.ethereum_client.lock_tokens.return_value = "0xethereum_lock_tx"
        self.bitcoin_client.release_tokens.return_value = "0xbitcoin_release_tx"
        
        # Create transfer request
        transfer_request = {
            'from_chain': ChainType.ETHEREUM,
            'to_chain': ChainType.BITCOIN,
            'amount': 1000000000000000000,
            'from_address': "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            'to_address': "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"
        }
        
        result = self.bridge.execute_cross_chain_transfer(transfer_request)
        
        self.assertIn('lock_tx_id', result)
        self.assertIn('release_tx_id', result)
        self.assertEqual(result['lock_tx_id'], "0xethereum_lock_tx")
        self.assertEqual(result['release_tx_id'], "0xbitcoin_release_tx")
    
    def test_route_optimization(self):
        """Test route optimization."""
        # Mock route calculation
        routes = [
            {
                'path': [ChainType.ETHEREUM, ChainType.POLYGON, ChainType.BSC],
                'total_fee': 0.01,
                'estimated_time': 300
            },
            {
                'path': [ChainType.ETHEREUM, ChainType.BSC],
                'total_fee': 0.02,
                'estimated_time': 600
            }
        ]
        
        best_route = self.bridge.find_optimal_route(
            ChainType.ETHEREUM,
            ChainType.BSC,
            1000000000000000000
        )
        
        self.assertIsInstance(best_route, dict)
        self.assertIn('path', best_route)
        self.assertIn('total_fee', best_route)
        self.assertIn('estimated_time', best_route)
    
    def test_fraud_detection(self):
        """Test fraud detection."""
        # Mock suspicious transaction
        suspicious_tx = UniversalTransaction(
            tx_id="0xsuspicious_tx",
            from_address="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            to_address="0x8ba1f109551bD432803012645Hac136c",
            amount=1000000000000000000,
            token_type=TokenType.NATIVE,
            chain_type=ChainType.ETHEREUM,
            created_at=time.time()
        )
        
        # Mock fraud detection
        fraud_score = self.bridge.detect_fraud(suspicious_tx)
        
        self.assertIsInstance(fraud_score, float)
        self.assertGreaterEqual(fraud_score, 0.0)
        self.assertLessEqual(fraud_score, 1.0)


class TestBridgeValidatorNetworkIntegration(unittest.TestCase):
    """Test bridge validator network integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ValidatorConfig(
            min_validators=3,
            max_validators=10,
            consensus_threshold=0.67,
            slashing_percentage=0.1
        )
        self.validator_network = BridgeValidatorNetwork(self.config)
    
    def test_network_initialization(self):
        """Test validator network initialization."""
        self.assertIsInstance(self.validator_network, BridgeValidatorNetwork)
        self.assertIsInstance(self.validator_network.config, ValidatorConfig)
    
    def test_validator_registration(self):
        """Test validator registration."""
        validator = Validator(
            validator_id="validator_1",
            public_key="0x1234567890abcdef",
            stake_amount=1000000000000000000,
            is_active=True
        )
        
        self.validator_network.register_validator(validator)
        
        self.assertIn("validator_1", self.validator_network.validators)
        self.assertEqual(self.validator_network.validators["validator_1"], validator)
    
    def test_consensus_mechanism(self):
        """Test BFT consensus mechanism."""
        # Register multiple validators
        validators = [
            Validator("validator_1", "0x1111", 1000000000000000000, True),
            Validator("validator_2", "0x2222", 1000000000000000000, True),
            Validator("validator_3", "0x3333", 1000000000000000000, True),
            Validator("validator_4", "0x4444", 1000000000000000000, True)
        ]
        
        for validator in validators:
            self.validator_network.register_validator(validator)
        
        # Test consensus on a proposal
        proposal = {
            'proposal_id': 'proposal_1',
            'transaction_id': '0x1234567890abcdef',
            'action': 'approve_transfer',
            'timestamp': time.time()
        }
        
        # Mock validator votes
        votes = {
            'validator_1': True,
            'validator_2': True,
            'validator_3': False,
            'validator_4': True
        }
        
        consensus_result = self.validator_network.reach_consensus(proposal, votes)
        
        self.assertIsInstance(consensus_result, dict)
        self.assertIn('approved', consensus_result)
        self.assertIn('consensus_reached', consensus_result)
        self.assertTrue(consensus_result['consensus_reached'])
        self.assertTrue(consensus_result['approved'])
    
    def test_slashing_mechanism(self):
        """Test slashing mechanism for malicious validators."""
        validator = Validator(
            validator_id="malicious_validator",
            public_key="0xmalicious",
            stake_amount=1000000000000000000,
            is_active=True
        )
        
        self.validator_network.register_validator(validator)
        
        # Test slashing
        slashing_result = self.validator_network.slash_validator(
            "malicious_validator",
            "double_spending"
        )
        
        self.assertIsInstance(slashing_result, dict)
        self.assertIn('slashed_amount', slashing_result)
        self.assertIn('reason', slashing_result)
        self.assertEqual(slashing_result['reason'], "double_spending")
    
    def test_emergency_pause(self):
        """Test emergency pause mechanism."""
        # Test emergency pause
        pause_result = self.validator_network.emergency_pause("security_breach")
        
        self.assertIsInstance(pause_result, dict)
        self.assertIn('paused', pause_result)
        self.assertIn('reason', pause_result)
        self.assertTrue(pause_result['paused'])
        self.assertEqual(pause_result['reason'], "security_breach")
        
        # Test emergency unpause
        unpause_result = self.validator_network.emergency_unpause()
        
        self.assertIsInstance(unpause_result, dict)
        self.assertIn('paused', unpause_result)
        self.assertFalse(unpause_result['paused'])


class TestAtomicSwapIntegration(unittest.TestCase):
    """Test atomic swap integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.swap_manager = AtomicSwapManager()
    
    def test_swap_proposal_creation(self):
        """Test atomic swap proposal creation."""
        proposal = SwapProposal(
            swap_id="swap_123",
            initiator="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            counterparty="0x8ba1f109551bD432803012645Hac136c",
            initiator_chain=ChainType.ETHEREUM,
            counterparty_chain=ChainType.BITCOIN,
            initiator_amount=1000000000000000000,
            counterparty_amount=0.05,
            timeout_duration=3600,
            status=SwapStatus.PENDING
        )
        
        self.assertIsInstance(proposal, SwapProposal)
        self.assertEqual(proposal.swap_id, "swap_123")
        self.assertEqual(proposal.status, SwapStatus.PENDING)
    
    def test_swap_execution(self):
        """Test atomic swap execution."""
        proposal = SwapProposal(
            swap_id="swap_456",
            initiator="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            counterparty="0x8ba1f109551bD432803012645Hac136c",
            initiator_chain=ChainType.ETHEREUM,
            counterparty_chain=ChainType.BITCOIN,
            initiator_amount=1000000000000000000,
            counterparty_amount=0.05,
            timeout_duration=3600,
            status=SwapStatus.PENDING
        )
        
        # Mock swap execution
        execution_result = self.swap_manager.execute_swap(proposal)
        
        self.assertIsInstance(execution_result, dict)
        self.assertIn('success', execution_result)
        self.assertIn('initiator_tx_id', execution_result)
        self.assertIn('counterparty_tx_id', execution_result)
    
    def test_swap_timeout(self):
        """Test atomic swap timeout handling."""
        proposal = SwapProposal(
            swap_id="swap_789",
            initiator="0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            counterparty="0x8ba1f109551bD432803012645Hac136c",
            initiator_chain=ChainType.ETHEREUM,
            counterparty_chain=ChainType.BITCOIN,
            initiator_amount=1000000000000000000,
            counterparty_amount=0.05,
            timeout_duration=1,  # 1 second timeout
            status=SwapStatus.PENDING
        )
        
        # Test timeout handling
        timeout_result = self.swap_manager.handle_timeout(proposal)
        
        self.assertIsInstance(timeout_result, dict)
        self.assertIn('timed_out', timeout_result)
        self.assertIn('refund_tx_id', timeout_result)


class TestCrossChainIntegration(unittest.TestCase):
    """Test cross-chain integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.universal_bridge = UniversalBridge()
        self.validator_network = BridgeValidatorNetwork(ValidatorConfig())
        
        # Mock chain clients
        self.ethereum_client = Mock()
        self.bitcoin_client = Mock()
        self.polygon_client = Mock()
        self.bsc_client = Mock()
        
        # Register clients
        self.universal_bridge.register_chain(ChainType.ETHEREUM, self.ethereum_client)
        self.universal_bridge.register_chain(ChainType.BITCOIN, self.bitcoin_client)
        self.universal_bridge.register_chain(ChainType.POLYGON, self.polygon_client)
        self.universal_bridge.register_chain(ChainType.BSC, self.bsc_client)
    
    def test_ethereum_to_bitcoin_transfer(self):
        """Test Ethereum to Bitcoin transfer."""
        # Mock chain operations
        self.ethereum_client.lock_tokens.return_value = "0xethereum_lock_tx"
        self.bitcoin_client.release_tokens.return_value = "0xbitcoin_release_tx"
        
        # Create transfer
        transfer_request = {
            'from_chain': ChainType.ETHEREUM,
            'to_chain': ChainType.BITCOIN,
            'amount': 1000000000000000000,  # 1 ETH
            'from_address': "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            'to_address': "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"
        }
        
        result = self.universal_bridge.execute_cross_chain_transfer(transfer_request)
        
        self.assertIn('lock_tx_id', result)
        self.assertIn('release_tx_id', result)
        self.ethereum_client.lock_tokens.assert_called_once()
        self.bitcoin_client.release_tokens.assert_called_once()
    
    def test_polygon_to_bsc_transfer(self):
        """Test Polygon to BSC transfer."""
        # Mock chain operations
        self.polygon_client.lock_tokens.return_value = "0xpolygon_lock_tx"
        self.bsc_client.release_tokens.return_value = "0xbsc_release_tx"
        
        # Create transfer
        transfer_request = {
            'from_chain': ChainType.POLYGON,
            'to_chain': ChainType.BSC,
            'amount': 1000000000000000000,  # 1 MATIC
            'from_address': "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            'to_address': "0x8ba1f109551bD432803012645Hac136c"
        }
        
        result = self.universal_bridge.execute_cross_chain_transfer(transfer_request)
        
        self.assertIn('lock_tx_id', result)
        self.assertIn('release_tx_id', result)
        self.polygon_client.lock_tokens.assert_called_once()
        self.bsc_client.release_tokens.assert_called_once()
    
    def test_multi_hop_transfer(self):
        """Test multi-hop transfer (Ethereum -> Polygon -> BSC)."""
        # Mock chain operations
        self.ethereum_client.lock_tokens.return_value = "0xethereum_lock_tx"
        self.polygon_client.release_tokens.return_value = "0xpolygon_release_tx"
        self.polygon_client.lock_tokens.return_value = "0xpolygon_lock_tx"
        self.bsc_client.release_tokens.return_value = "0xbsc_release_tx"
        
        # Create multi-hop transfer
        transfer_request = {
            'from_chain': ChainType.ETHEREUM,
            'to_chain': ChainType.BSC,
            'amount': 1000000000000000000,  # 1 ETH
            'from_address': "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            'to_address': "0x8ba1f109551bD432803012645Hac136c",
            'intermediate_chains': [ChainType.POLYGON]
        }
        
        result = self.universal_bridge.execute_cross_chain_transfer(transfer_request)
        
        self.assertIn('lock_tx_id', result)
        self.assertIn('release_tx_id', result)
        self.assertIn('intermediate_txs', result)
        
        # Verify all operations were called
        self.ethereum_client.lock_tokens.assert_called_once()
        self.polygon_client.release_tokens.assert_called_once()
        self.polygon_client.lock_tokens.assert_called_once()
        self.bsc_client.release_tokens.assert_called_once()
    
    def test_transfer_with_validator_consensus(self):
        """Test transfer with validator consensus."""
        # Register validators
        validators = [
            Validator("validator_1", "0x1111", 1000000000000000000, True),
            Validator("validator_2", "0x2222", 1000000000000000000, True),
            Validator("validator_3", "0x3333", 1000000000000000000, True)
        ]
        
        for validator in validators:
            self.validator_network.register_validator(validator)
        
        # Mock chain operations
        self.ethereum_client.lock_tokens.return_value = "0xethereum_lock_tx"
        self.bitcoin_client.release_tokens.return_value = "0xbitcoin_release_tx"
        
        # Create transfer with validator consensus
        transfer_request = {
            'from_chain': ChainType.ETHEREUM,
            'to_chain': ChainType.BITCOIN,
            'amount': 1000000000000000000,
            'from_address': "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            'to_address': "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",
            'require_consensus': True
        }
        
        result = self.universal_bridge.execute_cross_chain_transfer(transfer_request)
        
        self.assertIn('lock_tx_id', result)
        self.assertIn('release_tx_id', result)
        self.assertIn('consensus_result', result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
