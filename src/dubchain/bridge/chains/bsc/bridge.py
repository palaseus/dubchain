"""
BSC Bridge Implementation with BEP-20 and BEP-721 Support

This module provides comprehensive BSC bridging capabilities including:
- BEP-20 token bridging
- BEP-721 NFT bridging
- Fast block confirmation
- Gas optimization for BSC
- Bridge security and validation
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
import secrets

try:
    from web3 import Web3
    from web3.exceptions import ContractLogicError, Web3Exception
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

from ....errors import BridgeError, ClientError
from ....logging import get_logger
from .client import BSCClient, BSCConfig, BSCTransaction

logger = get_logger(__name__)


@dataclass
class BEP20Token:
    """BEP-20 token data."""
    address: str
    name: str
    symbol: str
    decimals: int
    total_supply: int
    balance: int = 0


@dataclass
class BEP721Token:
    """BEP-721 NFT data."""
    address: str
    name: str
    symbol: str
    token_id: int
    owner: str
    token_uri: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BridgeConfig:
    """Configuration for BSC bridge."""
    client_config: BSCConfig
    min_transfer_amount: int = 1000000000000000000  # 1 BNB
    max_transfer_amount: int = 1000000000000000000000  # 1000 BNB
    bridge_fee_percent: float = 0.1  # 0.1%
    enable_bep20_bridging: bool = True
    enable_bep721_bridging: bool = True
    enable_fast_confirmation: bool = True
    confirmation_blocks: int = 3
    enable_batch_processing: bool = True
    batch_size: int = 50


@dataclass
class BridgeTransaction:
    """BSC bridge transaction data."""
    tx_id: str
    from_address: str
    to_address: str
    amount: int
    token_address: Optional[str] = None  # None for BNB, address for BEP-20
    token_type: str = "BNB"  # BNB, BEP-20, BEP-721
    fee: int = 0
    confirmations: int = 0
    block_height: Optional[int] = None
    raw_transaction: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, confirmed, failed


class BEP20Manager:
    """Manages BEP-20 token operations."""
    
    def __init__(self, client: BSCClient):
        self.client = client
        
        # Standard BEP-20 ABI
        self.bep20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_spender", "type": "address"},
                    {"name": "_value", "type": "uint256"}
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [
                    {"name": "_owner", "type": "address"},
                    {"name": "_spender", "type": "address"}
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            }
        ]
    
    def get_token_info(self, token_address: str) -> Optional[BEP20Token]:
        """Get BEP-20 token information."""
        try:
            # Get token details
            name = self.client.call_contract(token_address, self.bep20_abi, "name")
            symbol = self.client.call_contract(token_address, self.bep20_abi, "symbol")
            decimals = self.client.call_contract(token_address, self.bep20_abi, "decimals")
            total_supply = self.client.call_contract(token_address, self.bep20_abi, "totalSupply")
            
            if not all([name, symbol, decimals is not None, total_supply is not None]):
                return None
            
            return BEP20Token(
                address=token_address,
                name=name,
                symbol=symbol,
                decimals=decimals,
                total_supply=total_supply
            )
            
        except Exception as e:
            logger.error(f"Failed to get token info for {token_address}: {e}")
            return None
    
    def get_token_balance(self, token_address: str, owner_address: str) -> int:
        """Get BEP-20 token balance."""
        try:
            balance = self.client.call_contract(token_address, self.bep20_abi, "balanceOf", [owner_address])
            return balance if balance is not None else 0
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            return 0
    
    def transfer_token(self, token_address: str, from_address: str, to_address: str, 
                      amount: int, private_key: str) -> str:
        """Transfer BEP-20 tokens."""
        try:
            # Build transfer transaction
            contract = self.client.web3.eth.contract(address=token_address, abi=self.bep20_abi)
            transfer_function = contract.functions.transfer(to_address, amount)
            
            # Get gas estimate
            gas_estimate = self.client.estimate_gas({
                'from': from_address,
                'to': token_address,
                'data': transfer_function._encode_transaction_data()
            })
            
            # Build transaction
            transaction = transfer_function.build_transaction({
                'from': from_address,
                'gas': gas_estimate,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address)
            })
            
            # Sign and send transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to transfer token: {e}")
            raise BridgeError(f"Token transfer failed: {e}")


class BEP721Manager:
    """Manages BEP-721 NFT operations."""
    
    def __init__(self, client: BSCClient):
        self.client = client
        
        # Standard BEP-721 ABI
        self.bep721_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_tokenId", "type": "uint256"}],
                "name": "ownerOf",
                "outputs": [{"name": "", "type": "address"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "name",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "symbol",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [{"name": "_tokenId", "type": "uint256"}],
                "name": "tokenURI",
                "outputs": [{"name": "", "type": "string"}],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_tokenId", "type": "uint256"}
                ],
                "name": "transferFrom",
                "outputs": [],
                "type": "function"
            },
            {
                "constant": False,
                "inputs": [
                    {"name": "_to", "type": "address"},
                    {"name": "_tokenId", "type": "uint256"}
                ],
                "name": "safeTransferFrom",
                "outputs": [],
                "type": "function"
            }
        ]
    
    def get_nft_info(self, nft_address: str, token_id: int) -> Optional[BEP721Token]:
        """Get BEP-721 NFT information."""
        try:
            # Get NFT details
            owner = self.client.call_contract(nft_address, self.bep721_abi, "ownerOf", [token_id])
            name = self.client.call_contract(nft_address, self.bep721_abi, "name")
            symbol = self.client.call_contract(nft_address, self.bep721_abi, "symbol")
            token_uri = self.client.call_contract(nft_address, self.bep721_abi, "tokenURI", [token_id])
            
            if not all([owner, name, symbol]):
                return None
            
            return BEP721Token(
                address=nft_address,
                name=name,
                symbol=symbol,
                token_id=token_id,
                owner=owner,
                token_uri=token_uri
            )
            
        except Exception as e:
            logger.error(f"Failed to get NFT info for {nft_address}:{token_id}: {e}")
            return None
    
    def transfer_nft(self, nft_address: str, from_address: str, to_address: str,
                    token_id: int, private_key: str) -> str:
        """Transfer BEP-721 NFT."""
        try:
            # Build transfer transaction
            contract = self.client.web3.eth.contract(address=nft_address, abi=self.bep721_abi)
            transfer_function = contract.functions.safeTransferFrom(from_address, to_address, token_id)
            
            # Get gas estimate
            gas_estimate = self.client.estimate_gas({
                'from': from_address,
                'to': nft_address,
                'data': transfer_function._encode_transaction_data()
            })
            
            # Build transaction
            transaction = transfer_function.build_transaction({
                'from': from_address,
                'gas': gas_estimate,
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address)
            })
            
            # Sign and send transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to transfer NFT: {e}")
            raise BridgeError(f"NFT transfer failed: {e}")


class BSCBridge:
    """Main BSC bridge implementation."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.client = BSCClient(config.client_config)
        self.bep20_manager = BEP20Manager(self.client)
        self.bep721_manager = BEP721Manager(self.client)
        self.pending_transactions: Dict[str, BridgeTransaction] = {}
        self._running = False
        
    async def start(self) -> None:
        """Start the BSC bridge."""
        if self._running:
            return
            
        self._running = True
        if self.config.enable_fast_confirmation:
            await self.client.start_confirmation_checker()
        logger.info("BSC bridge started")
        
    async def stop(self) -> None:
        """Stop the BSC bridge."""
        self._running = False
        if self.config.enable_fast_confirmation:
            await self.client.stop_confirmation_checker()
        logger.info("BSC bridge stopped")
    
    async def send_bnb(self, from_address: str, to_address: str, amount: int,
                      private_key: str) -> str:
        """Send BNB transaction."""
        try:
            # Validate amount
            if amount < self.config.min_transfer_amount:
                raise BridgeError(f"Amount too small. Minimum: {self.config.min_transfer_amount}")
            
            if amount > self.config.max_transfer_amount:
                raise BridgeError(f"Amount too large. Maximum: {self.config.max_transfer_amount}")
            
            # Calculate fee
            fee = self._calculate_fee(amount)
            total_amount = amount + fee
            
            # Check balance
            balance = self.client.get_balance(from_address)
            if balance < total_amount:
                raise BridgeError("Insufficient BNB balance")
            
            # Build transaction
            transaction = {
                'from': from_address,
                'to': to_address,
                'value': amount,
                'gas': self.client.estimate_gas({
                    'from': from_address,
                    'to': to_address,
                    'value': amount
                }),
                'gasPrice': self.client.get_optimized_gas_price(),
                'nonce': self.client.get_nonce(from_address)
            }
            
            # Sign and send transaction
            signed_txn = self.client.web3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.client.send_transaction(signed_txn.rawTransaction)
            
            # Store transaction
            bridge_tx = BridgeTransaction(
                tx_id=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                token_type="BNB",
                fee=fee,
                status="pending"
            )
            
            self.pending_transactions[tx_hash] = bridge_tx
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to send BNB: {e}")
            raise BridgeError(f"BNB transfer failed: {e}")
    
    async def send_bep20_token(self, token_address: str, from_address: str, 
                              to_address: str, amount: int, private_key: str) -> str:
        """Send BEP-20 token."""
        try:
            if not self.config.enable_bep20_bridging:
                raise BridgeError("BEP-20 bridging not enabled")
            
            # Get token info
            token_info = self.bep20_manager.get_token_info(token_address)
            if not token_info:
                raise BridgeError(f"Invalid BEP-20 token: {token_address}")
            
            # Check token balance
            balance = self.bep20_manager.get_token_balance(token_address, from_address)
            if balance < amount:
                raise BridgeError(f"Insufficient {token_info.symbol} balance")
            
            # Transfer token
            tx_hash = self.bep20_manager.transfer_token(
                token_address, from_address, to_address, amount, private_key
            )
            
            # Store transaction
            bridge_tx = BridgeTransaction(
                tx_id=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=amount,
                token_address=token_address,
                token_type="BEP-20",
                status="pending"
            )
            
            self.pending_transactions[tx_hash] = bridge_tx
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to send BEP-20 token: {e}")
            raise BridgeError(f"BEP-20 transfer failed: {e}")
    
    async def send_bep721_nft(self, nft_address: str, from_address: str,
                            to_address: str, token_id: int, private_key: str) -> str:
        """Send BEP-721 NFT."""
        try:
            if not self.config.enable_bep721_bridging:
                raise BridgeError("BEP-721 bridging not enabled")
            
            # Get NFT info
            nft_info = self.bep721_manager.get_nft_info(nft_address, token_id)
            if not nft_info:
                raise BridgeError(f"Invalid BEP-721 NFT: {nft_address}:{token_id}")
            
            # Check ownership
            if nft_info.owner.lower() != from_address.lower():
                raise BridgeError("Not the owner of this NFT")
            
            # Transfer NFT
            tx_hash = self.bep721_manager.transfer_nft(
                nft_address, from_address, to_address, token_id, private_key
            )
            
            # Store transaction
            bridge_tx = BridgeTransaction(
                tx_id=tx_hash,
                from_address=from_address,
                to_address=to_address,
                amount=1,  # NFTs are always 1
                token_address=nft_address,
                token_type="BEP-721",
                status="pending"
            )
            
            self.pending_transactions[tx_hash] = bridge_tx
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to send BEP-721 NFT: {e}")
            raise BridgeError(f"BEP-721 transfer failed: {e}")
    
    def _calculate_fee(self, amount: int) -> int:
        """Calculate bridge fee."""
        return int(amount * self.config.bridge_fee_percent / 100)
    
    async def get_transaction_status(self, tx_id: str) -> Optional[BridgeTransaction]:
        """Get transaction status."""
        if tx_id in self.pending_transactions:
            bridge_tx = self.pending_transactions[tx_id]
            
            # Update from blockchain
            tx_info = self.client.get_transaction(tx_id)
            if tx_info:
                bridge_tx.confirmations = tx_info.confirmations or 0
                bridge_tx.block_height = tx_info.block_number
                
                if bridge_tx.confirmations >= self.config.confirmation_blocks:
                    bridge_tx.status = "confirmed"
                elif bridge_tx.confirmations > 0:
                    bridge_tx.status = "pending"
            
            return bridge_tx
        
        return None
    
    async def batch_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Batch multiple transactions."""
        if not self.config.enable_batch_processing:
            raise BridgeError("Batch processing not enabled")
        
        try:
            # Group transactions by type
            bnb_txs = [tx for tx in transactions if tx.get("token_type") == "BNB"]
            bep20_txs = [tx for tx in transactions if tx.get("token_type") == "BEP-20"]
            bep721_txs = [tx for tx in transactions if tx.get("token_type") == "BEP-721"]
            
            batch_results = []
            
            # Process BNB transactions
            if bnb_txs:
                bnb_batch = await self._batch_bnb_transactions(bnb_txs)
                batch_results.append(bnb_batch)
            
            # Process BEP-20 transactions
            if bep20_txs:
                bep20_batch = await self._batch_bep20_transactions(bep20_txs)
                batch_results.append(bep20_batch)
            
            # Process BEP-721 transactions
            if bep721_txs:
                bep721_batch = await self._batch_bep721_transactions(bep721_txs)
                batch_results.append(bep721_batch)
            
            return f"batch_{len(batch_results)}_groups"
            
        except Exception as e:
            logger.error(f"Failed to batch transactions: {e}")
            raise BridgeError(f"Transaction batching failed: {e}")
    
    async def _batch_bnb_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Batch BNB transactions."""
        try:
            # This would involve creating a single transaction with multiple outputs
            # For now, just return a batch ID
            return f"bnb_batch_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Failed to batch BNB transactions: {e}")
            raise
    
    async def _batch_bep20_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Batch BEP-20 transactions."""
        try:
            # This would involve creating a single transaction with multiple token transfers
            # For now, just return a batch ID
            return f"bep20_batch_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Failed to batch BEP-20 transactions: {e}")
            raise
    
    async def _batch_bep721_transactions(self, transactions: List[Dict[str, Any]]) -> str:
        """Batch BEP-721 transactions."""
        try:
            # This would involve creating a single transaction with multiple NFT transfers
            # For now, just return a batch ID
            return f"bep721_batch_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Failed to batch BEP-721 transactions: {e}")
            raise
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        pending_count = len([tx for tx in self.pending_transactions.values() if tx.status == "pending"])
        confirmed_count = len([tx for tx in self.pending_transactions.values() if tx.status == "confirmed"])
        
        # Count by token type
        bnb_count = len([tx for tx in self.pending_transactions.values() if tx.token_type == "BNB"])
        bep20_count = len([tx for tx in self.pending_transactions.values() if tx.token_type == "BEP-20"])
        bep721_count = len([tx for tx in self.pending_transactions.values() if tx.token_type == "BEP-721"])
        
        return {
            "pending_transactions": pending_count,
            "confirmed_transactions": confirmed_count,
            "total_transactions": len(self.pending_transactions),
            "bnb_transactions": bnb_count,
            "bep20_transactions": bep20_count,
            "bep721_transactions": bep721_count,
            "bep20_bridging_enabled": self.config.enable_bep20_bridging,
            "bep721_bridging_enabled": self.config.enable_bep721_bridging,
            "fast_confirmation_enabled": self.config.enable_fast_confirmation,
            "batch_processing_enabled": self.config.enable_batch_processing,
            "running": self._running
        }
