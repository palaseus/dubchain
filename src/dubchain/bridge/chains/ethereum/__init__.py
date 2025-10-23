"""
Ethereum Web3 client integration for DubChain.

This module provides comprehensive Ethereum blockchain integration including:
- Web3.py client with RPC communication
- Transaction management and monitoring
- Smart contract interaction
- Event monitoring and indexing
- Gas price optimization
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
    from eth_account import Account
    from eth_account.signers.local import LocalAccount
    ETH_ACCOUNT_AVAILABLE = True
except ImportError:
    ETH_ACCOUNT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .monitoring import (
    EthereumMonitoringService,
    MonitoringConfig,
    EventData,
    GasPriceData,
    GasPriceOracle as MonitoringGasPriceOracle,
    EventMonitor,
)
from .security import (
    EthereumBridgeSecurity,
    SecurityConfig,
    SecurityLevel,
    ThreatType,
    SecurityEvent,
    TransactionValidation,
    TransactionValidator,
    FraudDetector,
    RateLimiter,
    MultiSigValidator,
    EmergencyPauseManager,
)
from .bridge import (
    ProductionEthereumBridge,
    ProductionBridgeConfig,
    BridgeTransaction,
    BridgeMetrics,
    BridgeStatus,
    TransactionStatus,
)
from .contracts import (
    ERC20Contract,
    ERC721Contract,
    BridgeContract,
    ContractInfo,
)


@dataclass
class EthereumConfig:
    """Configuration for Ethereum client."""
    
    rpc_url: str = "http://localhost:8545"
    chain_id: int = 1  # Mainnet
    gas_price_strategy: str = "fast"  # slow, standard, fast, instant
    max_gas_price_gwei: int = 100
    gas_limit: int = 21000
    timeout_seconds: int = 30
    retry_count: int = 3
    enable_poa_middleware: bool = False
    enable_gas_price_oracle: bool = True
    gas_price_oracle_url: str = "https://api.etherscan.io/api"
    api_key: Optional[str] = None


@dataclass
class EthereumTransaction:
    """Ethereum transaction data."""
    
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


@dataclass
class EthereumBlock:
    """Ethereum block data."""
    
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


@dataclass
class GasPriceInfo:
    """Gas price information."""
    
    slow_gwei: float
    standard_gwei: float
    fast_gwei: float
    instant_gwei: float
    base_fee_gwei: Optional[float] = None
    priority_fee_gwei: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class GasPriceOracle:
    """Gas price oracle for Ethereum."""
    
    def __init__(self, config: EthereumConfig):
        """Initialize gas price oracle."""
        self.config = config
        self.cached_prices: Optional[GasPriceInfo] = None
        self.cache_duration = 60  # 1 minute cache
    
    def get_gas_prices(self) -> GasPriceInfo:
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
            logger.info(f"Failed to get gas prices: {e}")
            # Return default prices
            return GasPriceInfo(
                slow_gwei=20.0,
                standard_gwei=30.0,
                fast_gwei=50.0,
                instant_gwei=100.0,
            )
    
    def _fetch_from_oracle(self) -> GasPriceInfo:
        """Fetch gas prices from external oracle."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("Requests library not available")
        
        # Use Etherscan API for gas prices
        url = f"{self.config.gas_price_oracle_url}?module=gastracker&action=gasoracle&apikey={self.config.api_key or ''}"
        
        response = requests.get(url, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") != "1":
            raise RuntimeError(f"Oracle API error: {data.get('message', 'Unknown error')}")
        
        result = data["result"]
        
        return GasPriceInfo(
            slow_gwei=float(result["SafeGasPrice"]),
            standard_gwei=float(result["ProposeGasPrice"]),
            fast_gwei=float(result["FastGasPrice"]),
            instant_gwei=float(result["FastGasPrice"]) * 1.5,
            base_fee_gwei=float(result.get("suggestBaseFee", 0)),
            priority_fee_gwei=float(result.get("suggestPriorityFee", 0)),
        )
    
    def _estimate_gas_prices(self) -> GasPriceInfo:
        """Estimate gas prices using Web3."""
        # This is a simplified estimation
        # Real implementation would use more sophisticated methods
        return GasPriceInfo(
            slow_gwei=20.0,
            standard_gwei=30.0,
            fast_gwei=50.0,
            instant_gwei=100.0,
        )


class EthereumClient:
    """Ethereum Web3 client."""
    
    def __init__(self, config: EthereumConfig):
        """Initialize Ethereum client."""
        self.config = config
        self.web3: Optional[Web3] = None
        self.gas_oracle = GasPriceOracle(config)
        self.monitoring_service: Optional[EthereumMonitoringService] = None
        self._initialized = False
        
        if WEB3_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Web3 client."""
        try:
            # Create Web3 instance
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            # Add PoA middleware if needed
            if self.config.enable_poa_middleware:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Test connection
            if not self.web3.is_connected():
                raise RuntimeError("Failed to connect to Ethereum node")
            
            # Verify chain ID
            actual_chain_id = self.web3.eth.chain_id
            if actual_chain_id != self.config.chain_id:
                logger.info(f"Warning: Chain ID mismatch. Expected {self.config.chain_id}, got {actual_chain_id}")
            
            self._initialized = True
            
            # Initialize monitoring service
            if self.web3:
                monitoring_config = MonitoringConfig(
                    poll_interval=2.0,
                    gas_price_update_interval=10.0,
                    enable_eip1559=True,
                    enable_event_indexing=True
                )
                self.monitoring_service = EthereumMonitoringService(self.web3, monitoring_config)
            
        except Exception as e:
            logger.info(f"Failed to initialize Ethereum client: {e}")
            self._initialized = False
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._initialized and self.web3 and self.web3.is_connected()
    
    def test_connection(self) -> bool:
        """Test connection to Ethereum network."""
        return self.is_connected()
    
    def get_latest_block(self) -> Optional[EthereumBlock]:
        """Get latest block."""
        if not self.is_connected():
            return None
        
        try:
            block = self.web3.eth.get_block('latest')
            return self._parse_block(block)
        except Exception as e:
            logger.info(f"Failed to get latest block: {e}")
            return None
    
    def get_block(self, block_number: int) -> Optional[EthereumBlock]:
        """Get block by number."""
        if not self.is_connected():
            return None
        
        try:
            block = self.web3.eth.get_block(block_number)
            return self._parse_block(block)
        except BlockNotFound:
            return None
        except Exception as e:
            logger.info(f"Failed to get block {block_number}: {e}")
            return None
    
    def get_transaction(self, tx_hash: str) -> Optional[EthereumTransaction]:
        """Get transaction by hash."""
        if not self.is_connected():
            return None
        
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            return self._parse_transaction(tx)
        except TransactionNotFound:
            return None
        except Exception as e:
            logger.info(f"Failed to get transaction {tx_hash}: {e}")
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
            logger.info(f"Failed to get transaction receipt {tx_hash}: {e}")
            return None
    
    def get_balance(self, address: str) -> int:
        """Get ETH balance for address."""
        if not self.is_connected():
            return 0
        
        try:
            return self.web3.eth.get_balance(address)
        except Exception as e:
            logger.info(f"Failed to get balance for {address}: {e}")
            return 0
    
    def get_nonce(self, address: str) -> int:
        """Get nonce for address."""
        if not self.is_connected():
            return 0
        
        try:
            return self.web3.eth.get_transaction_count(address)
        except Exception as e:
            logger.info(f"Failed to get nonce for {address}: {e}")
            return 0
    
    def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for transaction."""
        if not self.is_connected():
            return self.config.gas_limit
        
        try:
            return self.web3.eth.estimate_gas(transaction)
        except Exception as e:
            logger.info(f"Failed to estimate gas: {e}")
            return self.config.gas_limit
    
    def get_gas_price(self) -> int:
        """Get current gas price."""
        if not self.is_connected():
            return 20 * 10**9  # 20 gwei
        
        try:
            return self.web3.eth.gas_price
        except Exception as e:
            logger.info(f"Failed to get gas price: {e}")
            return 20 * 10**9
    
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
        """Send transaction to network."""
        if not self.is_connected():
            raise RuntimeError("Ethereum client not connected")
        
        try:
            tx_hash = self.web3.eth.send_raw_transaction(transaction)
            return tx_hash.hex()
        except Exception as e:
            logger.info(f"Failed to send transaction: {e}")
            raise
    
    def wait_for_transaction(self, tx_hash: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
        """Wait for transaction to be mined."""
        if not self.is_connected():
            return None
        
        try:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            return dict(receipt)
        except Exception as e:
            logger.info(f"Failed to wait for transaction {tx_hash}: {e}")
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
            logger.info(f"Contract logic error: {e}")
            return None
        except Exception as e:
            logger.info(f"Failed to call contract function: {e}")
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
            logger.info(f"Failed to get contract events: {e}")
            return []
    
    def _parse_block(self, block: Any) -> EthereumBlock:
        """Parse Web3 block to EthereumBlock."""
        return EthereumBlock(
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
    
    def _parse_transaction(self, tx: Any) -> EthereumTransaction:
        """Parse Web3 transaction to EthereumTransaction."""
        return EthereumTransaction(
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
                "connected": True,
            }
        except Exception as e:
            logger.info(f"Failed to get network info: {e}")
            return {"connected": False, "error": str(e)}
    
    async def start_monitoring(self) -> None:
        """Start Ethereum monitoring service."""
        if self.monitoring_service:
            await self.monitoring_service.start()
    
    async def stop_monitoring(self) -> None:
        """Stop Ethereum monitoring service."""
        if self.monitoring_service:
            await self.monitoring_service.stop()
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        if self.monitoring_service:
            return self.monitoring_service.get_monitoring_stats()
        return {}
    
    def get_gas_price_trend(self, hours: int = 1) -> Dict[str, float]:
        """Get gas price trend."""
        if self.monitoring_service:
            return self.monitoring_service.get_gas_price_trend(hours)
        return {}
    
    def add_event_filter(self, filter_id: str, filter_params: Dict[str, Any]) -> None:
        """Add event filter for monitoring."""
        if self.monitoring_service:
            self.monitoring_service.add_event_filter(filter_id, filter_params)
    
    def remove_event_filter(self, filter_id: str) -> None:
        """Remove event filter."""
        if self.monitoring_service:
            self.monitoring_service.remove_event_filter(filter_id)
    
    def get_event_history(self, limit: Optional[int] = None) -> List[EventData]:
        """Get event history."""
        if self.monitoring_service:
            return self.monitoring_service.get_event_history(limit)
        return []
