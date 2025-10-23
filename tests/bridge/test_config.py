"""
Bridge Test Configuration

This module provides configuration and utilities for bridge integration tests.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import unittest
import os
import sys
from typing import Dict, List, Any, Optional
import tempfile
import json
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class BridgeTestConfig:
    """Configuration for bridge integration tests."""
    
    def __init__(self):
        self.enable_ethereum_tests = os.getenv('ENABLE_ETHEREUM_TESTS', 'true').lower() == 'true'
        self.enable_bitcoin_tests = os.getenv('ENABLE_BITCOIN_TESTS', 'true').lower() == 'true'
        self.enable_polygon_tests = os.getenv('ENABLE_POLYGON_TESTS', 'true').lower() == 'true'
        self.enable_bsc_tests = os.getenv('ENABLE_BSC_TESTS', 'true').lower() == 'true'
        self.enable_security_tests = os.getenv('ENABLE_SECURITY_TESTS', 'true').lower() == 'true'
        self.enable_penetration_tests = os.getenv('ENABLE_PENETRATION_TESTS', 'false').lower() == 'true'
        
        # Test network configurations
        self.ethereum_rpc_url = os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/test')
        self.bitcoin_rpc_host = os.getenv('BITCOIN_RPC_HOST', 'localhost')
        self.bitcoin_rpc_port = int(os.getenv('BITCOIN_RPC_PORT', '18332'))
        self.polygon_rpc_url = os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com')
        self.bsc_rpc_url = os.getenv('BSC_RPC_URL', 'https://bsc-dataseed.binance.org')
        
        # Test data directory
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Test timeouts
        self.transaction_timeout = int(os.getenv('TRANSACTION_TIMEOUT', '30'))
        self.consensus_timeout = int(os.getenv('CONSENSUS_TIMEOUT', '60'))
        self.swap_timeout = int(os.getenv('SWAP_TIMEOUT', '300'))


class BridgeTestUtils:
    """Utilities for bridge integration tests."""
    
    @staticmethod
    def create_test_transaction(chain_type: str, amount: int = 1000000000000000000) -> Dict[str, Any]:
        """Create a test transaction."""
        return {
            'tx_id': f'0x{chain_type}_test_tx',
            'from_address': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'to_address': '0x8ba1f109551bD432803012645Hac136c',
            'amount': amount,
            'chain_type': chain_type,
            'created_at': time.time()
        }
    
    @staticmethod
    def create_test_validator(validator_id: str, stake_amount: int = 1000000000000000000) -> Dict[str, Any]:
        """Create a test validator."""
        return {
            'validator_id': validator_id,
            'public_key': f'0x{validator_id}',
            'stake_amount': stake_amount,
            'is_active': True
        }
    
    @staticmethod
    def create_test_swap_proposal(swap_id: str) -> Dict[str, Any]:
        """Create a test swap proposal."""
        return {
            'swap_id': swap_id,
            'initiator': '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6',
            'counterparty': '0x8ba1f109551bD432803012645Hac136c',
            'initiator_chain': 'ethereum',
            'counterparty_chain': 'bitcoin',
            'initiator_amount': 1000000000000000000,
            'counterparty_amount': 0.05,
            'timeout_duration': 3600,
            'status': 'pending'
        }
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], filename: str) -> None:
        """Save test results to file."""
        config = BridgeTestConfig()
        filepath = os.path.join(config.test_data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    @staticmethod
    def load_test_results(filename: str) -> Optional[Dict[str, Any]]:
        """Load test results from file."""
        config = BridgeTestConfig()
        filepath = os.path.join(config.test_data_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def compare_test_results(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current test results with baseline."""
        comparison = {}
        
        for key in current:
            if key in baseline:
                current_val = current[key]
                baseline_val = baseline[key]
                
                if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                    if baseline_val != 0:
                        improvement = ((current_val - baseline_val) / baseline_val) * 100
                        comparison[key] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'improvement_percent': improvement
                        }
        
        return comparison


class BridgeTestFixtures:
    """Test fixtures for bridge integration tests."""
    
    def __init__(self):
        self.config = BridgeTestConfig()
        self.utils = BridgeTestUtils()
    
    def get_test_transactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get test transactions."""
        transactions = []
        for i in range(count):
            tx = self.utils.create_test_transaction(f'test_chain_{i}')
            transactions.append(tx)
        return transactions
    
    def get_test_validators(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get test validators."""
        validators = []
        for i in range(count):
            validator = self.utils.create_test_validator(f'validator_{i}')
            validators.append(validator)
        return validators
    
    def get_test_swap_proposals(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get test swap proposals."""
        proposals = []
        for i in range(count):
            proposal = self.utils.create_test_swap_proposal(f'swap_{i}')
            proposals.append(proposal)
        return proposals
    
    def get_test_network_topology(self) -> Dict[str, Any]:
        """Get test network topology."""
        return {
            'nodes': ['node_1', 'node_2', 'node_3', 'node_4', 'node_5'],
            'connections': [
                ('node_1', 'node_2'),
                ('node_1', 'node_3'),
                ('node_2', 'node_4'),
                ('node_3', 'node_5')
            ],
            'latencies': {
                ('node_1', 'node_2'): 10,
                ('node_1', 'node_3'): 15,
                ('node_2', 'node_4'): 20,
                ('node_3', 'node_5'): 25
            }
        }


# Global test configuration
test_config = BridgeTestConfig()
test_utils = BridgeTestUtils()
test_fixtures = BridgeTestFixtures()


def skip_if_chain_disabled(chain_name: str):
    """Skip test if chain is disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if chain_name.lower() == 'ethereum' and not test_config.enable_ethereum_tests:
                pytest.skip("Ethereum tests disabled")
            elif chain_name.lower() == 'bitcoin' and not test_config.enable_bitcoin_tests:
                pytest.skip("Bitcoin tests disabled")
            elif chain_name.lower() == 'polygon' and not test_config.enable_polygon_tests:
                pytest.skip("Polygon tests disabled")
            elif chain_name.lower() == 'bsc' and not test_config.enable_bsc_tests:
                pytest.skip("BSC tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_security_tests_disabled():
    """Skip test if security tests are disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not test_config.enable_security_tests:
                pytest.skip("Security tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def skip_if_penetration_tests_disabled():
    """Skip test if penetration tests are disabled."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not test_config.enable_penetration_tests:
                pytest.skip("Penetration tests disabled")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def async_test(func):
    """Decorator for async test functions."""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper
