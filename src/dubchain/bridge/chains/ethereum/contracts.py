"""
Ethereum smart contract management for DubChain bridge.

This module provides comprehensive smart contract interaction including:
- ERC-20 token contract management
- ERC-721 NFT contract management
- Bridge contract interaction
- Contract deployment and management
"""

import logging

logger = logging.getLogger(__name__)
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from web3 import Web3
    from web3.contract import Contract
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

from .client import EthereumClient, EthereumConfig


@dataclass
class ContractInfo:
    """Smart contract information."""
    
    address: str
    abi: List[Dict[str, Any]]
    bytecode: Optional[str] = None
    name: Optional[str] = None
    version: Optional[str] = None
    deployed_at: Optional[int] = None
    deployer: Optional[str] = None


class ERC20Contract:
    """ERC-20 token contract interface."""
    
    def __init__(self, client: EthereumClient, contract_info: ContractInfo):
        """Initialize ERC-20 contract."""
        self.client = client
        self.contract_info = contract_info
        self.contract: Optional[Contract] = None
        
        if WEB3_AVAILABLE and client.web3:
            self.contract = client.web3.eth.contract(
                address=contract_info.address,
                abi=contract_info.abi
            )
    
    def name(self) -> Optional[str]:
        """Get token name."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.name().call()
        except Exception as e:
            logger.info(f"Failed to get token name: {e}")
            return None
    
    def symbol(self) -> Optional[str]:
        """Get token symbol."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.symbol().call()
        except Exception as e:
            logger.info(f"Failed to get token symbol: {e}")
            return None
    
    def decimals(self) -> Optional[int]:
        """Get token decimals."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.decimals().call()
        except Exception as e:
            logger.info(f"Failed to get token decimals: {e}")
            return None
    
    def total_supply(self) -> Optional[int]:
        """Get total supply."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.totalSupply().call()
        except Exception as e:
            logger.info(f"Failed to get total supply: {e}")
            return None
    
    def balance_of(self, address: str) -> Optional[int]:
        """Get balance of address."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.balanceOf(address).call()
        except Exception as e:
            logger.info(f"Failed to get balance: {e}")
            return None
    
    def allowance(self, owner: str, spender: str) -> Optional[int]:
        """Get allowance."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.allowance(owner, spender).call()
        except Exception as e:
            logger.info(f"Failed to get allowance: {e}")
            return None
    
    def transfer(self, to: str, amount: int, from_address: str, private_key: str) -> Optional[str]:
        """Transfer tokens."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.transfer(to, amount).build_transaction({
                'from': from_address,
                'gas': 100000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to transfer tokens: {e}")
            return None
    
    def approve(self, spender: str, amount: int, from_address: str, private_key: str) -> Optional[str]:
        """Approve spender."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.approve(spender, amount).build_transaction({
                'from': from_address,
                'gas': 100000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to approve spender: {e}")
            return None
    
    def transfer_from(self, from_address: str, to: str, amount: int, spender: str, private_key: str) -> Optional[str]:
        """Transfer from address."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.transferFrom(from_address, to, amount).build_transaction({
                'from': spender,
                'gas': 100000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(spender),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to transfer from: {e}")
            return None


class ERC721Contract:
    """ERC-721 NFT contract interface."""
    
    def __init__(self, client: EthereumClient, contract_info: ContractInfo):
        """Initialize ERC-721 contract."""
        self.client = client
        self.contract_info = contract_info
        self.contract: Optional[Contract] = None
        
        if WEB3_AVAILABLE and client.web3:
            self.contract = client.web3.eth.contract(
                address=contract_info.address,
                abi=contract_info.abi
            )
    
    def name(self) -> Optional[str]:
        """Get contract name."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.name().call()
        except Exception as e:
            logger.info(f"Failed to get contract name: {e}")
            return None
    
    def symbol(self) -> Optional[str]:
        """Get contract symbol."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.symbol().call()
        except Exception as e:
            logger.info(f"Failed to get contract symbol: {e}")
            return None
    
    def total_supply(self) -> Optional[int]:
        """Get total supply."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.totalSupply().call()
        except Exception as e:
            logger.info(f"Failed to get total supply: {e}")
            return None
    
    def balance_of(self, owner: str) -> Optional[int]:
        """Get balance of owner."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.balanceOf(owner).call()
        except Exception as e:
            logger.info(f"Failed to get balance: {e}")
            return None
    
    def owner_of(self, token_id: int) -> Optional[str]:
        """Get owner of token."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.ownerOf(token_id).call()
        except Exception as e:
            logger.info(f"Failed to get owner of token: {e}")
            return None
    
    def token_uri(self, token_id: int) -> Optional[str]:
        """Get token URI."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.tokenURI(token_id).call()
        except Exception as e:
            logger.info(f"Failed to get token URI: {e}")
            return None
    
    def approve(self, to: str, token_id: int, from_address: str, private_key: str) -> Optional[str]:
        """Approve token."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.approve(to, token_id).build_transaction({
                'from': from_address,
                'gas': 100000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to approve token: {e}")
            return None
    
    def transfer_from(self, from_address: str, to: str, token_id: int, spender: str, private_key: str) -> Optional[str]:
        """Transfer token."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.transferFrom(from_address, to, token_id).build_transaction({
                'from': spender,
                'gas': 100000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(spender),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to transfer token: {e}")
            return None


class BridgeContract:
    """Bridge contract interface."""
    
    def __init__(self, client: EthereumClient, contract_info: ContractInfo):
        """Initialize bridge contract."""
        self.client = client
        self.contract_info = contract_info
        self.contract: Optional[Contract] = None
        
        if WEB3_AVAILABLE and client.web3:
            self.contract = client.web3.eth.contract(
                address=contract_info.address,
                abi=contract_info.abi
            )
    
    def lock_tokens(self, token_address: str, amount: int, from_address: str, private_key: str) -> Optional[str]:
        """Lock tokens for bridging."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.lockTokens(token_address, amount).build_transaction({
                'from': from_address,
                'gas': 200000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to lock tokens: {e}")
            return None
    
    def unlock_tokens(self, token_address: str, amount: int, to_address: str, from_address: str, private_key: str) -> Optional[str]:
        """Unlock tokens after bridging."""
        if not self.contract:
            return None
        
        try:
            # Build transaction
            transaction = self.contract.functions.unlockTokens(token_address, amount, to_address).build_transaction({
                'from': from_address,
                'gas': 200000,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address),
            })
            
            # Sign transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            return tx_hash
            
        except Exception as e:
            logger.info(f"Failed to unlock tokens: {e}")
            return None
    
    def get_locked_amount(self, token_address: str) -> Optional[int]:
        """Get locked amount for token."""
        if not self.contract:
            return None
        
        try:
            return self.contract.functions.getLockedAmount(token_address).call()
        except Exception as e:
            logger.info(f"Failed to get locked amount: {e}")
            return None


class ContractManager:
    """Manages smart contracts."""
    
    def __init__(self, client: EthereumClient):
        """Initialize contract manager."""
        self.client = client
        self.contracts: Dict[str, ContractInfo] = {}
        self.erc20_contracts: Dict[str, ERC20Contract] = {}
        self.erc721_contracts: Dict[str, ERC721Contract] = {}
        self.bridge_contracts: Dict[str, BridgeContract] = {}
    
    def add_contract(self, contract_info: ContractInfo) -> bool:
        """Add contract to manager."""
        try:
            self.contracts[contract_info.address] = contract_info
            
            # Create appropriate contract instance based on ABI
            if self._is_erc20_contract(contract_info.abi):
                self.erc20_contracts[contract_info.address] = ERC20Contract(self.client, contract_info)
            elif self._is_erc721_contract(contract_info.abi):
                self.erc721_contracts[contract_info.address] = ERC721Contract(self.client, contract_info)
            elif self._is_bridge_contract(contract_info.abi):
                self.bridge_contracts[contract_info.address] = BridgeContract(self.client, contract_info)
            
            return True
            
        except Exception as e:
            logger.info(f"Failed to add contract: {e}")
            return False
    
    def get_erc20_contract(self, address: str) -> Optional[ERC20Contract]:
        """Get ERC-20 contract."""
        return self.erc20_contracts.get(address)
    
    def get_erc721_contract(self, address: str) -> Optional[ERC721Contract]:
        """Get ERC-721 contract."""
        return self.erc721_contracts.get(address)
    
    def get_bridge_contract(self, address: str) -> Optional[BridgeContract]:
        """Get bridge contract."""
        return self.bridge_contracts.get(address)
    
    def _is_erc20_contract(self, abi: List[Dict[str, Any]]) -> bool:
        """Check if contract is ERC-20."""
        required_functions = ['name', 'symbol', 'decimals', 'totalSupply', 'balanceOf', 'transfer', 'approve']
        functions = [item['name'] for item in abi if item.get('type') == 'function']
        return all(func in functions for func in required_functions)
    
    def _is_erc721_contract(self, abi: List[Dict[str, Any]]) -> bool:
        """Check if contract is ERC-721."""
        required_functions = ['name', 'symbol', 'totalSupply', 'balanceOf', 'ownerOf', 'tokenURI', 'approve', 'transferFrom']
        functions = [item['name'] for item in abi if item.get('type') == 'function']
        return all(func in functions for func in required_functions)
    
    def _is_bridge_contract(self, abi: List[Dict[str, Any]]) -> bool:
        """Check if contract is bridge contract."""
        required_functions = ['lockTokens', 'unlockTokens', 'getLockedAmount']
        functions = [item['name'] for item in abi if item.get('type') == 'function']
        return all(func in functions for func in required_functions)
    
    def get_contract_info(self, address: str) -> Optional[ContractInfo]:
        """Get contract information."""
        return self.contracts.get(address)
    
    def list_contracts(self) -> List[ContractInfo]:
        """List all contracts."""
        return list(self.contracts.values())
    
    def remove_contract(self, address: str) -> bool:
        """Remove contract from manager."""
        try:
            if address in self.contracts:
                del self.contracts[address]
            
            if address in self.erc20_contracts:
                del self.erc20_contracts[address]
            
            if address in self.erc721_contracts:
                del self.erc721_contracts[address]
            
            if address in self.bridge_contracts:
                del self.bridge_contracts[address]
            
            return True
            
        except Exception as e:
            logger.info(f"Failed to remove contract: {e}")
            return False
