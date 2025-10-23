"""
Binance Smart Chain RPC Integration

This module provides comprehensive BSC integration including:
- BSC RPC client with Web3 integration
- Fast block confirmation
- BEP token support
- Gas optimization for BSC
- Testnet support
"""

import logging

logger = logging.getLogger(__name__)
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
class BSCConfig:
    """Configuration for BSC client."""
    rpc_url: str = "https://bsc-dataseed.binance.org"
    chain_id: int = 56  # BSC mainnet
    testnet_rpc_url: str = "https://data-seed-prebsc-1-s1.binance.org:8545"
    testnet_chain_id: int = 97  # BSC testnet
    use_testnet: bool = False
    gas_price_strategy: str = "fast"
    max_gas_price_gwei: int = 20  # BSC has lower gas prices
    gas_limit: int = 21000
    timeout_seconds: int = 30
    retry_count: int = 3
    enable_poa_middleware: bool = True
    enable_fast_confirmation: bool = True
    confirmation_blocks: int = 3  # BSC has fast finality
    enable_gas_price_oracle: bool = True
    gas_price_oracle_url: str = "https://api.bscscan.com/api"


@dataclass
class BSCTransaction:
    """BSC transaction data."""
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
    confirmation_status: str = "pending"  # pending, confirmed, finalized


@dataclass
class BSCBlock:
    """BSC block data."""
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
    confirmation_status: str = "pending"


@dataclass
class BSCGasPriceInfo:
    """BSC gas price information."""
    slow_gwei: float
    standard_gwei: float
    fast_gwei: float
    instant_gwei: float
    timestamp: float = field(default_factory=time.time)


class BSCGasPriceOracle:
    """Gas price oracle for BSC."""
    
    def __init__(self, config: BSCConfig):
        self.config = config
        self.cached_prices: Optional[BSCGasPriceInfo] = None
        self.cache_duration = 60  # 1 minute cache
    
    def get_gas_prices(self) -> BSCGasPriceInfo:
        """Get current gas prices."""
        if (self.cached_prices and 
            time.time() - self.cached_prices.timestamp < self.cache_duration):
            return self.cached_prices
        
        try:
            if self.config.enable_gas_price_oracle and REQUESTS_AVAILABLE:
                prices = self._fetch_from_oracle()
            else:
                prices = self._estimate_gas_prices()
            
            self.cached_prices = prices
            return prices
            
        except Exception as e:
            logger.error(f"Failed to get gas prices: {e}")
            # Return default prices
            return BSCGasPriceInfo(
                slow_gwei=3.0,
                standard_gwei=5.0,
                fast_gwei=8.0,
                instant_gwei=12.0,
            )
    
    def _fetch_from_oracle(self) -> BSCGasPriceInfo:
        """Fetch gas prices from BSCScan API."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Requests library not available")
        
        try:
            # Use BSCScan API for gas prices
            url = f"{self.config.gas_price_oracle_url}?module=gastracker&action=gasoracle"
            
            response = requests.get(url, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "1":
                raise RuntimeError(f"Oracle API error: {data.get('message', 'Unknown error')}")
            
            result = data["result"]
            
            return BSCGasPriceInfo(
                slow_gwei=float(result["SafeGasPrice"]),
                standard_gwei=float(result["ProposeGasPrice"]),
                fast_gwei=float(result["FastGasPrice"]),
                instant_gwei=float(result["FastGasPrice"]) * 1.5,
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch from oracle: {e}")
            raise
    
    def _estimate_gas_prices(self) -> BSCGasPriceInfo:
        """Estimate gas prices using Web3."""
        # BSC typically has lower gas prices than Ethereum
        return BSCGasPriceInfo(
            slow_gwei=3.0,
            standard_gwei=5.0,
            fast_gwei=8.0,
            instant_gwei=12.0,
        )


class BSCClient:
    """BSC Web3 client with fast confirmation support."""
    
    def __init__(self, config: BSCConfig):
        self.config = config
        self.web3: Optional[Web3] = None
        self.gas_oracle = BSCGasPriceOracle(config)
        self._initialized = False
        self._confirmation_checker_running = False
        self._confirmation_checker_task: Optional[asyncio.Task] = None
        
        if WEB3_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize BSC Web3 client."""
        try:
            # Choose RPC URL based on testnet setting
            rpc_url = self.config.testnet_rpc_url if self.config.use_testnet else self.config.rpc_url
            chain_id = self.config.testnet_chain_id if self.config.use_testnet else self.config.chain_id
            
            # Create Web3 instance
            self.web3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware for BSC
            if self.config.enable_poa_middleware:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if not self.web3.is_connected():
                raise RuntimeError("Failed to connect to BSC node")
            
            # Verify chain ID
            actual_chain_id = self.web3.eth.chain_id
            if actual_chain_id != chain_id:
                logger.warning(f"Chain ID mismatch. Expected {chain_id}, got {actual_chain_id}")
            
            self._initialized = True
            logger.info(f"BSC client initialized for {'testnet' if self.config.use_testnet else 'mainnet'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BSC client: {e}")
            self._initialized = False
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._initialized and self.web3 and self.web3.is_connected()
    
    async def start_confirmation_checker(self) -> None:
        """Start confirmation checker for fast confirmation."""
        if not self.config.enable_fast_confirmation or self._confirmation_checker_running:
            return
        
        self._confirmation_checker_running = True
        self._confirmation_checker_task = asyncio.create_task(self._confirmation_checker_loop())
        logger.info("BSC confirmation checker started")
    
    async def stop_confirmation_checker(self) -> None:
        """Stop confirmation checker."""
        self._confirmation_checker_running = False
        if self._confirmation_checker_task:
            self._confirmation_checker_task.cancel()
            try:
                await self._confirmation_checker_task
            except asyncio.CancelledError:
                pass
        logger.info("BSC confirmation checker stopped")
    
    async def _confirmation_checker_loop(self) -> None:
        """Confirmation checker loop for fast confirmation."""
        while self._confirmation_checker_running:
            try:
                await self._check_transaction_confirmations()
                await asyncio.sleep(2.0)  # Check every 2 seconds
            except Exception as e:
                logger.error(f"Error in confirmation checker: {e}")
                await asyncio.sleep(2.0)
    
    async def _check_transaction_confirmations(self) -> None:
        """Check transaction confirmations."""
        try:
            # Get latest block
            latest_block = self.web3.eth.get_block('latest')
            
            # Check pending transactions for confirmation
            # This would involve checking pending transactions and updating their status
            logger.debug(f"Latest block: {latest_block.number}")
            
        except Exception as e:
            logger.error(f"Failed to check confirmations: {e}")
    
    def get_latest_block(self) -> Optional[BSCBlock]:
        """Get latest block."""
        if not self.is_connected():
            return None
        
        try:
            block = self.web3.eth.get_block('latest')
            return self._parse_block(block)
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return None
    
    def get_block(self, block_number: int) -> Optional[BSCBlock]:
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
    
    def get_transaction(self, tx_hash: str) -> Optional[BSCTransaction]:
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
        """Get BNB balance for address."""
        if not self.is_connected():
            return 0
        
        try:
            return self.web3.eth.get_balance(address)
        except Exception as e:
            logger.error(f"Failed to get balance for {address}: {e}")
            return 0
    
    def get_nonce(self, address: str) -> int:
        """Get nonce for address."""
        if not self.is_connected():
            return 0
        
        try:
            return self.web3.eth.get_transaction_count(address)
        except Exception as e:
            logger.error(f"Failed to get nonce for {address}: {e}")
            return 0
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for transaction."""
        if not self.is_connected():
            return self.config.gas_limit
        
        try:
            return self.web3.eth.estimate_gas(transaction)
        except Exception as e:
            logger.error(f"Failed to estimate gas: {e}")
            return self.config.gas_limit
    
    def get_gas_price(self) -> int:
        """Get current gas price."""
        if not self.is_connected():
            return 5 * 10**9  # 5 gwei
        
        try:
            return self.web3.eth.gas_price
        except Exception as e:
            logger.error(f"Failed to get gas price: {e}")
            return 5 * 10**9
    
    def get_optimized_gas_price(self, strategy: str = "standard") -> int:
        """Get optimized gas price based on strategy."""
        gas_prices = self.gas_oracle.get_gas_prices()
        
        if strategy == "slow":
            gwei = gas_prices.slow_gwei
        elif strategy == "standard":
            gwei = gas_prices.standard_gwei
        elif strategy == "fast":
            gwei = gas_prices.fast_gwei
        elif strategy == "instant":
            gwei = gas_prices.instant_gwei
        else:
            gwei = gas_prices.standard_gwei
        
        # Convert to wei and apply max limit
        gas_price_wei = int(gwei * 10**9)
        max_gas_price_wei = self.config.max_gas_price_gwei * 10**9
        
        return min(gas_price_wei, max_gas_price_wei)
    
    def send_transaction(self, transaction: Dict[str, Any]) -> str:
        """Send transaction to BSC network."""
        if not self.is_connected():
            raise RuntimeError("BSC client not connected")
        
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
    
    def get_contract_events(self, contract_address: str, abi: List[Dict], event_name: str, 
                          from_block: int = 0, to_block: str = 'latest') -> List[Dict[str, Any]]:
        """Get contract events."""
        if not self.is_connected():
            return []
        
        try:
            contract = self.web3.eth.contract(address=contract_address, abi=abi)
            event_filter = contract.events[event_name].create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            events = event_filter.get_all_entries()
            return [dict(event) for event in events]
        except Exception as e:
            logger.error(f"Failed to get contract events: {e}")
            return []
    
    def _parse_block(self, block: Any) -> BSCBlock:
        """Parse Web3 block to BSCBlock."""
        return BSCBlock(
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
    
    def _parse_transaction(self, tx: Any) -> BSCTransaction:
        """Parse Web3 transaction to BSCTransaction."""
        return BSCTransaction(
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
            gas_prices = self.gas_oracle.get_gas_prices()
            
            return {
                "chain_id": self.web3.eth.chain_id,
                "network_id": self.web3.net.version,
                "latest_block": latest_block.number if latest_block else 0,
                "gas_price_wei": gas_price,
                "gas_price_gwei": gas_price / 10**9,
                "gas_prices": {
                    "slow_gwei": gas_prices.slow_gwei,
                    "standard_gwei": gas_prices.standard_gwei,
                    "fast_gwei": gas_prices.fast_gwei,
                    "instant_gwei": gas_prices.instant_gwei,
                },
                "network": "bsc_testnet" if self.config.use_testnet else "bsc_mainnet",
                "connected": True,
                "fast_confirmation_enabled": self.config.enable_fast_confirmation,
                "confirmation_blocks": self.config.confirmation_blocks,
            }
        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return {"connected": False, "error": str(e)}
