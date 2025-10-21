"""
Polygon RPC Integration and PoS Bridge

This module provides comprehensive Polygon network integration including:
- Polygon RPC client with Web3 integration
- PoS bridge functionality
- Fast finality optimizations
- Gas optimization for Polygon network
- Mumbai testnet support
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal

try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    from web3.exceptions import (
        BlockNotFound,
        TransactionNotFound,
        ContractLogicError,
        Web3Exception,
    )
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from ....errors import BridgeError, ClientError
from ....logging import get_logger

logger = get_logger(__name__)


@dataclass
class PolygonConfig:
    """Configuration for Polygon client."""
    rpc_url: str = "https://polygon-rpc.com"
    chain_id: int = 137  # Polygon mainnet
    testnet_rpc_url: str = "https://rpc-mumbai.maticvigil.com"
    testnet_chain_id: int = 80001  # Mumbai testnet
    use_testnet: bool = False
    gas_price_strategy: str = "fast"
    max_gas_price_gwei: int = 50  # Polygon has lower gas prices
    gas_limit: int = 21000
    timeout_seconds: int = 30
    retry_count: int = 3
    enable_poa_middleware: bool = True
    enable_fast_finality: bool = True
    finality_check_interval: float = 2.0  # seconds
    max_finality_wait: int = 30  # seconds


@dataclass
class PolygonTransaction:
    """Polygon transaction data."""
    hash: str
    from_address: str
    to_address: str
    value: int
    gas: int
    gas_price: int
    nonce: int
    data: str
    block_number: Optional[int] = None
    block_hash: Optional[str] = None
    transaction_index: Optional[int] = None
    status: Optional[int] = None
    receipt: Optional[Dict[str, Any]] = None
    finality_status: str = "pending"  # pending, finalized, failed


@dataclass
class PolygonBlock:
    """Polygon block data."""
    number: int
    hash: str
    parent_hash: str
    timestamp: int
    gas_limit: int
    gas_used: int
    transactions: List[str]
    miner: str
    difficulty: int
    total_difficulty: int
    size: int
    base_fee_per_gas: Optional[int] = None
    finality_status: str = "pending"


@dataclass
class PoSBridgeConfig:
    """Configuration for Polygon PoS bridge."""
    ethereum_contract_address: str = "0x8484Ef722627bf18ca5Ae6BcF031c23E6e922B30"
    polygon_contract_address: str = "0x8484Ef722627bf18ca5Ae6BcF031c23E6e922B30"
    checkpoint_manager: str = "0x86E4Dc95c7FBdB52e52d419D6550dc974540c0af"
    fx_root: str = "0xfe5e5D361b2ad62c541bAb87C45a0C9fda55a43f"
    fx_child: str = "0x8397259c983751DAf40400790063935a11afa28a"
    min_deposit_amount: int = 1000000000000000000  # 1 ETH
    max_deposit_amount: int = 1000000000000000000000  # 1000 ETH
    bridge_fee_percent: float = 0.1  # 0.1%
    enable_auto_withdraw: bool = True
    withdrawal_delay: int = 30  # minutes


class PolygonClient:
    """Polygon Web3 client with fast finality support."""
    
    def __init__(self, config: PolygonConfig):
        self.config = config
        self.web3: Optional[Web3] = None
        self._initialized = False
        self._finality_checker_running = False
        self._finality_checker_task: Optional[asyncio.Task] = None
        
        if WEB3_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Polygon Web3 client."""
        try:
            # Choose RPC URL based on testnet setting
            rpc_url = self.config.testnet_rpc_url if self.config.use_testnet else self.config.rpc_url
            chain_id = self.config.testnet_chain_id if self.config.use_testnet else self.config.chain_id
            
            # Create Web3 instance
            self.web3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware for Polygon
            if self.config.enable_poa_middleware:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if not self.web3.is_connected():
                raise RuntimeError("Failed to connect to Polygon node")
            
            # Verify chain ID
            actual_chain_id = self.web3.eth.chain_id
            if actual_chain_id != chain_id:
                logger.warning(f"Chain ID mismatch. Expected {chain_id}, got {actual_chain_id}")
            
            self._initialized = True
            logger.info(f"Polygon client initialized for {'testnet' if self.config.use_testnet else 'mainnet'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            self._initialized = False
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._initialized and self.web3 and self.web3.is_connected()
    
    async def start_finality_checker(self) -> None:
        """Start finality checker for fast finality."""
        if not self.config.enable_fast_finality or self._finality_checker_running:
            return
        
        self._finality_checker_running = True
        self._finality_checker_task = asyncio.create_task(self._finality_checker_loop())
        logger.info("Polygon finality checker started")
    
    async def stop_finality_checker(self) -> None:
        """Stop finality checker."""
        self._finality_checker_running = False
        if self._finality_checker_task:
            self._finality_checker_task.cancel()
            try:
                await self._finality_checker_task
            except asyncio.CancelledError:
                pass
        logger.info("Polygon finality checker stopped")
    
    async def _finality_checker_loop(self) -> None:
        """Finality checker loop for fast finality."""
        while self._finality_checker_running:
            try:
                await self._check_block_finality()
                await asyncio.sleep(self.config.finality_check_interval)
            except Exception as e:
                logger.error(f"Error in finality checker: {e}")
                await asyncio.sleep(self.config.finality_check_interval)
    
    async def _check_block_finality(self) -> None:
        """Check block finality status."""
        try:
            # Get latest block
            latest_block = self.web3.eth.get_block('latest')
            
            # Check if block is finalized (simplified check)
            # In reality, this would involve checking with checkpoint manager
            if latest_block.number > 0:
                # Mark blocks as finalized after a certain number of confirmations
                finalized_block = latest_block.number - 32  # 32 blocks for Polygon
                logger.debug(f"Block {finalized_block} is considered finalized")
                
        except Exception as e:
            logger.error(f"Failed to check block finality: {e}")
    
    def get_latest_block(self) -> Optional[PolygonBlock]:
        """Get latest block."""
        if not self.is_connected():
            return None
        
        try:
            block = self.web3.eth.get_block('latest')
            return self._parse_block(block)
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return None
    
    def get_block(self, block_number: int) -> Optional[PolygonBlock]:
        """Get block by number."""
        if not self.is_connected():
            return None
        
        try:
            block = self.web3.eth.get_block(block_number)
            return self._parse_block(block)
        except BlockNotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to get block {block_number}: {e}")
            return None
    
    def get_transaction(self, tx_hash: str) -> Optional[PolygonTransaction]:
        """Get transaction by hash."""
        if not self.is_connected():
            return None
        
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            return self._parse_transaction(tx)
        except TransactionNotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to get transaction {tx_hash}: {e}")
            return None
    
    def get_transaction_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction receipt."""
        if not self.is_connected():
            return None
        
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except TransactionNotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to get transaction receipt {tx_hash}: {e}")
            return None
    
    def get_balance(self, address: str) -> int:
        """Get MATIC balance for address."""
        if not self.is_connected():
            return 0
        
        try:
            return self.web3.eth.get_balance(address)
        except Exception as e:
            logger.error(f"Failed to get balance for {address}: {e}")
            return 0
    
    def get_gas_price(self) -> int:
        """Get current gas price."""
        if not self.is_connected():
            return 20 * 10**9  # 20 gwei
        
        try:
            return self.web3.eth.gas_price
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return 20 * 10**9
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for transaction."""
        if not self.is_connected():
            return self.config.gas_limit
        
        try:
            return self.web3.eth.estimate_gas(transaction)
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            return self.config.gas_limit
    
    def send_transaction(self, transaction: Dict[str, Any]) -> str:
        """Send transaction to Polygon network."""
        if not self.is_connected():
            raise RuntimeError("Polygon client not connected")
        
        try:
            tx_hash = self.web3.eth.send_raw_transaction(transaction)
            return tx_hash.hex()
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            raise
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
        """Wait for transaction to be mined."""
        if not self.is_connected():
            return None
        
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            return dict(receipt)
        except Exception as e:
            logger.error(f"Failed to wait for transaction {tx_hash}: {e}")
            return None
    
    def call_contract(self, contract_address: str, abi: List[Dict], function_name: str, args: List[Any] = None) -> Any:
        """Call contract function."""
        if not self.is_connected():
            return None
        
        try:
            contract = self.web3.eth.contract(address=contract_address, abi=abi)
            function = getattr(contract.functions, function_name)
            
            if args:
                result = function(*args).call()
            else:
                result = function().call()
            
            return result
        except ContractLogicError as e:
            logger.error(f"Contract logic error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to call contract function: {e}")
            return None
    
    def _parse_block(self, block: Any) -> PolygonBlock:
        """Parse Web3 block to PolygonBlock."""
        return PolygonBlock(
            number=block.number,
            hash=block.hash.hex(),
            parent_hash=block.parentHash.hex(),
            timestamp=block.timestamp,
            gas_limit=block.gasLimit,
            gas_used=block.gasUsed,
            transactions=[tx.hex() if hasattr(tx, 'hex') else str(tx) for tx in block.transactions],
            miner=block.miner,
            difficulty=block.difficulty,
            total_difficulty=block.totalDifficulty,
            size=block.size,
            base_fee_per_gas=getattr(block, 'baseFeePerGas', None),
        )
    
    def _parse_transaction(self, tx: Any) -> PolygonTransaction:
        """Parse Web3 transaction to PolygonTransaction."""
        return PolygonTransaction(
            hash=tx.hash.hex(),
            from_address=tx['from'],
            to_address=tx.to,
            value=tx.value,
            gas=tx.gas,
            gas_price=tx.gasPrice,
            nonce=tx.nonce,
            data=tx.input.hex(),
            block_number=getattr(tx, 'blockNumber', None),
            block_hash=tx.blockHash.hex() if tx.blockHash else None,
            transaction_index=getattr(tx, 'transactionIndex', None),
        )
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        if not self.is_connected():
            return {}
        
        try:
            latest_block = self.get_latest_block()
            gas_price = self.get_gas_price()
            
            return {
                "chain_id": self.web3.eth.chain_id,
                "network_id": self.web3.net.version,
                "latest_block": latest_block.number if latest_block else 0,
                "gas_price_wei": gas_price,
                "gas_price_gwei": gas_price / 10**9,
                "network": "polygon_testnet" if self.config.use_testnet else "polygon_mainnet",
                "connected": True,
                "fast_finality_enabled": self.config.enable_fast_finality,
            }
        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return {"connected": False, "error": str(e)}


class PoSBridge:
    """Polygon PoS bridge implementation."""
    
    def __init__(self, polygon_config: PolygonConfig, bridge_config: PoSBridgeConfig):
        self.polygon_config = polygon_config
        self.bridge_config = bridge_config
        self.polygon_client = PolygonClient(polygon_config)
        self.pending_deposits: Dict[str, Dict[str, Any]] = {}
        self.pending_withdrawals: Dict[str, Dict[str, Any]] = {}
        self._running = False
    
    async def start(self) -> None:
        """Start the PoS bridge."""
        if self._running:
            return
        
        self._running = True
        if self.polygon_config.enable_fast_finality:
            await self.polygon_client.start_finality_checker()
        logger.info("Polygon PoS bridge started")
    
    async def stop(self) -> None:
        """Stop the PoS bridge."""
        self._running = False
        if self.polygon_config.enable_fast_finality:
            await self.polygon_client.stop_finality_checker()
        logger.info("Polygon PoS bridge stopped")
    
    async def deposit_to_polygon(self, ethereum_tx_hash: str, amount: int, 
                               recipient_address: str) -> str:
        """Deposit from Ethereum to Polygon."""
        try:
            # Validate deposit amount
            if amount < self.bridge_config.min_deposit_amount:
                raise BridgeError(f"Deposit amount too small. Minimum: {self.bridge_config.min_deposit_amount}")
            
            if amount > self.bridge_config.max_deposit_amount:
                raise BridgeError(f"Deposit amount too large. Maximum: {self.bridge_config.max_deposit_amount}")
            
            # Calculate bridge fee
            bridge_fee = int(amount * self.bridge_config.bridge_fee_percent / 100)
            net_amount = amount - bridge_fee
            
            # Create deposit record
            deposit_id = f"deposit_{ethereum_tx_hash}_{int(time.time())}"
            deposit_record = {
                "id": deposit_id,
                "ethereum_tx_hash": ethereum_tx_hash,
                "amount": amount,
                "bridge_fee": bridge_fee,
                "net_amount": net_amount,
                "recipient_address": recipient_address,
                "status": "pending",
                "created_at": time.time(),
                "polygon_tx_hash": None
            }
            
            self.pending_deposits[deposit_id] = deposit_record
            
            # Process deposit (simplified - in reality would involve checkpoint verification)
            await self._process_deposit(deposit_id)
            
            return deposit_id
            
        except Exception as e:
            logger.error(f"Failed to deposit to Polygon: {e}")
            raise BridgeError(f"Deposit failed: {e}")
    
    async def withdraw_to_ethereum(self, polygon_tx_hash: str, amount: int,
                                  recipient_address: str) -> str:
        """Withdraw from Polygon to Ethereum."""
        try:
            # Validate withdrawal amount
            if amount < self.bridge_config.min_deposit_amount:
                raise BridgeError(f"Withdrawal amount too small. Minimum: {self.bridge_config.min_deposit_amount}")
            
            # Calculate bridge fee
            bridge_fee = int(amount * self.bridge_config.bridge_fee_percent / 100)
            net_amount = amount - bridge_fee
            
            # Create withdrawal record
            withdrawal_id = f"withdrawal_{polygon_tx_hash}_{int(time.time())}"
            withdrawal_record = {
                "id": withdrawal_id,
                "polygon_tx_hash": polygon_tx_hash,
                "amount": amount,
                "bridge_fee": bridge_fee,
                "net_amount": net_amount,
                "recipient_address": recipient_address,
                "status": "pending",
                "created_at": time.time(),
                "ethereum_tx_hash": None
            }
            
            self.pending_withdrawals[withdrawal_id] = withdrawal_record
            
            # Process withdrawal (simplified - in reality would involve checkpoint submission)
            await self._process_withdrawal(withdrawal_id)
            
            return withdrawal_id
            
        except Exception as e:
            logger.error(f"Failed to withdraw to Ethereum: {e}")
            raise BridgeError(f"Withdrawal failed: {e}")
    
    async def _process_deposit(self, deposit_id: str) -> None:
        """Process deposit from Ethereum to Polygon."""
        try:
            deposit_record = self.pending_deposits[deposit_id]
            
            # In reality, this would involve:
            # 1. Verifying the Ethereum transaction
            # 2. Waiting for checkpoint confirmation
            # 3. Creating Polygon transaction
            
            # Simulate processing
            await asyncio.sleep(1)
            
            # Create Polygon transaction
            polygon_tx_hash = await self._create_polygon_transaction(
                deposit_record["recipient_address"],
                deposit_record["net_amount"]
            )
            
            deposit_record["polygon_tx_hash"] = polygon_tx_hash
            deposit_record["status"] = "completed"
            
            logger.info(f"Deposit {deposit_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process deposit {deposit_id}: {e}")
            self.pending_deposits[deposit_id]["status"] = "failed"
    
    async def _process_withdrawal(self, withdrawal_id: str) -> None:
        """Process withdrawal from Polygon to Ethereum."""
        try:
            withdrawal_record = self.pending_withdrawals[withdrawal_id]
            
            # In reality, this would involve:
            # 1. Verifying the Polygon transaction
            # 2. Submitting checkpoint to Ethereum
            # 3. Creating Ethereum transaction
            
            # Simulate processing
            await asyncio.sleep(1)
            
            # Create Ethereum transaction (simplified)
            ethereum_tx_hash = f"eth_tx_{withdrawal_id}"
            
            withdrawal_record["ethereum_tx_hash"] = ethereum_tx_hash
            withdrawal_record["status"] = "completed"
            
            logger.info(f"Withdrawal {withdrawal_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process withdrawal {withdrawal_id}: {e}")
            self.pending_withdrawals[withdrawal_id]["status"] = "failed"
    
    async def _create_polygon_transaction(self, recipient_address: str, amount: int) -> str:
        """Create Polygon transaction."""
        try:
            # This would involve creating a real transaction
            # For now, return a simulated transaction hash
            return f"polygon_tx_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Failed to create Polygon transaction: {e}")
            raise
    
    def get_deposit_status(self, deposit_id: str) -> Optional[Dict[str, Any]]:
        """Get deposit status."""
        return self.pending_deposits.get(deposit_id)
    
    def get_withdrawal_status(self, withdrawal_id: str) -> Optional[Dict[str, Any]]:
        """Get withdrawal status."""
        return self.pending_withdrawals.get(withdrawal_id)
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        pending_deposits = len([d for d in self.pending_deposits.values() if d["status"] == "pending"])
        completed_deposits = len([d for d in self.pending_deposits.values() if d["status"] == "completed"])
        pending_withdrawals = len([w for w in self.pending_withdrawals.values() if w["status"] == "pending"])
        completed_withdrawals = len([w for w in self.pending_withdrawals.values() if w["status"] == "completed"])
        
        return {
            "pending_deposits": pending_deposits,
            "completed_deposits": completed_deposits,
            "pending_withdrawals": pending_withdrawals,
            "completed_withdrawals": completed_withdrawals,
            "total_deposits": len(self.pending_deposits),
            "total_withdrawals": len(self.pending_withdrawals),
            "bridge_fee_percent": self.bridge_config.bridge_fee_percent,
            "min_deposit_amount": self.bridge_config.min_deposit_amount,
            "max_deposit_amount": self.bridge_config.max_deposit_amount,
            "running": self._running
        }
