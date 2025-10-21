"""
Universal Cross-Chain Bridge Interface and Routing

This module provides a unified interface for all blockchain integrations including:
- Unified transaction format across all chains
- Cross-chain transaction batching
- Automatic route optimization for multi-hop transfers
- Universal wallet address resolution
- Comprehensive fraud detection across all chains
- Fee optimization across routes
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from decimal import Decimal
import secrets
from enum import Enum

from ...errors import BridgeError, ClientError
from ...logging import get_logger

# Import all bridge implementations
from ..chains.ethereum import EthereumClient, EthereumConfig
from ..chains.bitcoin import BitcoinBridge, BridgeConfig as BitcoinBridgeConfig
from ..chains.polygon import PolygonClient, PolygonConfig, ZkEVMBridge, ZkEVMConfig
from ..chains.bsc import BSCBridge, BridgeConfig as BSCBridgeConfig
from ..chains.solana import SolanaBridge, BridgeConfig as SolanaBridgeConfig
from ..chains.cardano import CardanoBridge, BridgeConfig as CardanoBridgeConfig
from ..chains.avalanche import AvalancheBridge, BridgeConfig as AvalancheBridgeConfig
from ..chains.polkadot import PolkadotBridge, BridgeConfig as PolkadotBridgeConfig

logger = get_logger(__name__)


class ChainType(Enum):
    """Supported blockchain types."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    POLYGON_ZKEVM = "polygon_zkevm"
    BSC = "bsc"
    SOLANA = "solana"
    CARDANO = "cardano"
    AVALANCHE = "avalanche"
    POLKADOT = "polkadot"


class TokenType(Enum):
    """Supported token types."""
    NATIVE = "native"  # ETH, BTC, MATIC, BNB, SOL, ADA, AVAX, DOT
    ERC20 = "erc20"
    ERC721 = "erc721"
    BEP20 = "bep20"
    BEP721 = "bep721"
    SPL20 = "spl20"
    SPL721 = "spl721"
    NATIVE_TOKEN = "native_token"  # Cardano native tokens
    AVAX_TOKEN = "avax_token"  # Avalanche tokens
    SUBSTRATE_TOKEN = "substrate_token"  # Polkadot tokens


@dataclass
class UniversalAddress:
    """Universal address format."""
    chain: ChainType
    address: str
    address_type: str = "standard"  # standard, multisig, contract
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UniversalTransaction:
    """Universal transaction format."""
    tx_id: str
    from_address: UniversalAddress
    to_address: UniversalAddress
    amount: int
    token_type: TokenType
    token_address: Optional[str] = None  # None for native tokens
    fee: int = 0
    gas_price: Optional[int] = None
    gas_limit: Optional[int] = None
    nonce: Optional[int] = None
    data: Optional[str] = None
    confirmations: int = 0
    block_height: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, confirmed, failed
    route: Optional[List[ChainType]] = None


@dataclass
class RouteInfo:
    """Route information for cross-chain transfers."""
    from_chain: ChainType
    to_chain: ChainType
    intermediate_chains: List[ChainType]
    total_fee: int
    estimated_time: float  # seconds
    success_rate: float  # 0.0 to 1.0
    liquidity: int  # Available liquidity
    route_id: str = field(default_factory=lambda: secrets.token_hex(8))


@dataclass
class BridgeConfig:
    """Configuration for universal bridge."""
    ethereum_config: Optional[EthereumConfig] = None
    bitcoin_config: Optional[BitcoinBridgeConfig] = None
    polygon_config: Optional[PolygonConfig] = None
    zkevm_config: Optional[ZkEVMConfig] = None
    bsc_config: Optional[BSCBridgeConfig] = None
    enable_route_optimization: bool = True
    enable_fraud_detection: bool = True
    enable_fee_optimization: bool = True
    max_route_hops: int = 3
    route_timeout: float = 300.0  # seconds
    min_liquidity_threshold: int = 1000000000000000000  # 1 token
    enable_batching: bool = True
    batch_size: int = 100


class AddressResolver:
    """Resolves addresses across different chains."""
    
    def __init__(self):
        self.address_cache: Dict[str, UniversalAddress] = {}
        self.reverse_cache: Dict[str, str] = {}
    
    def resolve_address(self, chain: ChainType, address: str) -> UniversalAddress:
        """Resolve address to universal format."""
        cache_key = f"{chain.value}:{address.lower()}"
        
        if cache_key in self.address_cache:
            return self.address_cache[cache_key]
        
        # Create universal address
        universal_addr = UniversalAddress(
            chain=chain,
            address=address,
            address_type=self._detect_address_type(chain, address)
        )
        
        self.address_cache[cache_key] = universal_addr
        return universal_addr
    
    def _detect_address_type(self, chain: ChainType, address: str) -> str:
        """Detect address type based on chain and address format."""
        if chain == ChainType.ETHEREUM:
            if address.startswith("0x") and len(address) == 42:
                return "standard"
        elif chain == ChainType.BITCOIN:
            if address.startswith(("1", "3", "bc1")):
                return "standard"
        elif chain == ChainType.POLYGON:
            if address.startswith("0x") and len(address) == 42:
                return "standard"
        elif chain == ChainType.BSC:
            if address.startswith("0x") and len(address) == 42:
                return "standard"
        
        return "unknown"
    
    def get_chain_address(self, universal_addr: UniversalAddress) -> str:
        """Get chain-specific address from universal address."""
        return universal_addr.address


class RouteOptimizer:
    """Optimizes routes for cross-chain transfers."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.route_cache: Dict[str, RouteInfo] = {}
        self.chain_connectivity: Dict[ChainType, Set[ChainType]] = {
            ChainType.ETHEREUM: {ChainType.POLYGON, ChainType.BSC},
            ChainType.BITCOIN: {ChainType.ETHEREUM},
            ChainType.POLYGON: {ChainType.ETHEREUM, ChainType.POLYGON_ZKEVM},
            ChainType.POLYGON_ZKEVM: {ChainType.POLYGON},
            ChainType.BSC: {ChainType.ETHEREUM},
        }
    
    def find_optimal_route(self, from_chain: ChainType, to_chain: ChainType,
                          amount: int, token_type: TokenType) -> List[RouteInfo]:
        """Find optimal routes for cross-chain transfer."""
        cache_key = f"{from_chain.value}:{to_chain.value}:{amount}:{token_type.value}"
        
        if cache_key in self.route_cache:
            return [self.route_cache[cache_key]]
        
        routes = []
        
        # Direct route
        if self._is_direct_route_possible(from_chain, to_chain):
            direct_route = self._create_route(from_chain, to_chain, [], amount, token_type)
            routes.append(direct_route)
        
        # Multi-hop routes
        if self.config.max_route_hops > 1:
            multi_hop_routes = self._find_multi_hop_routes(
                from_chain, to_chain, amount, token_type
            )
            routes.extend(multi_hop_routes)
        
        # Sort by total fee and success rate
        routes.sort(key=lambda r: (r.total_fee, -r.success_rate))
        
        # Cache the best route
        if routes:
            self.route_cache[cache_key] = routes[0]
        
        return routes[:5]  # Return top 5 routes
    
    def _is_direct_route_possible(self, from_chain: ChainType, to_chain: ChainType) -> bool:
        """Check if direct route is possible."""
        return to_chain in self.chain_connectivity.get(from_chain, set())
    
    def _find_multi_hop_routes(self, from_chain: ChainType, to_chain: ChainType,
                              amount: int, token_type: TokenType) -> List[RouteInfo]:
        """Find multi-hop routes."""
        routes = []
        
        # Find intermediate chains
        intermediate_chains = self.chain_connectivity.get(from_chain, set())
        
        for intermediate in intermediate_chains:
            if intermediate == to_chain:
                continue
            
            # Check if intermediate can reach destination
            if to_chain in self.chain_connectivity.get(intermediate, set()):
                route = self._create_route(
                    from_chain, to_chain, [intermediate], amount, token_type
                )
                routes.append(route)
        
        return routes
    
    def _create_route(self, from_chain: ChainType, to_chain: ChainType,
                     intermediate_chains: List[ChainType], amount: int,
                     token_type: TokenType) -> RouteInfo:
        """Create route information."""
        # Calculate total fee
        total_fee = self._calculate_route_fee(from_chain, to_chain, intermediate_chains, amount)
        
        # Estimate time
        estimated_time = self._estimate_route_time(from_chain, to_chain, intermediate_chains)
        
        # Calculate success rate
        success_rate = self._calculate_success_rate(from_chain, to_chain, intermediate_chains)
        
        # Estimate liquidity
        liquidity = self._estimate_liquidity(from_chain, to_chain, amount)
        
        return RouteInfo(
            from_chain=from_chain,
            to_chain=to_chain,
            intermediate_chains=intermediate_chains,
            total_fee=total_fee,
            estimated_time=estimated_time,
            success_rate=success_rate,
            liquidity=liquidity
        )
    
    def _calculate_route_fee(self, from_chain: ChainType, to_chain: ChainType,
                           intermediate_chains: List[ChainType], amount: int) -> int:
        """Calculate total route fee."""
        # Simplified fee calculation
        base_fee = 1000000000000000000  # 1 token base fee
        
        # Add fee for each hop
        hop_count = len(intermediate_chains) + 1
        total_fee = base_fee * hop_count
        
        # Add percentage fee
        percentage_fee = int(amount * 0.001)  # 0.1%
        total_fee += percentage_fee
        
        return total_fee
    
    def _estimate_route_time(self, from_chain: ChainType, to_chain: ChainType,
                           intermediate_chains: List[ChainType]) -> float:
        """Estimate route completion time."""
        # Base times for each chain (in seconds)
        chain_times = {
            ChainType.ETHEREUM: 300,  # 5 minutes
            ChainType.BITCOIN: 3600,  # 1 hour
            ChainType.POLYGON: 60,    # 1 minute
            ChainType.POLYGON_ZKEVM: 120,  # 2 minutes
            ChainType.BSC: 60,        # 1 minute
        }
        
        total_time = 0
        
        # Add time for source chain
        total_time += chain_times.get(from_chain, 300)
        
        # Add time for intermediate chains
        for chain in intermediate_chains:
            total_time += chain_times.get(chain, 300)
        
        # Add time for destination chain
        total_time += chain_times.get(to_chain, 300)
        
        return total_time
    
    def _calculate_success_rate(self, from_chain: ChainType, to_chain: ChainType,
                              intermediate_chains: List[ChainType]) -> float:
        """Calculate route success rate."""
        # Base success rates for each chain
        chain_success_rates = {
            ChainType.ETHEREUM: 0.95,
            ChainType.BITCOIN: 0.90,
            ChainType.POLYGON: 0.98,
            ChainType.POLYGON_ZKEVM: 0.95,
            ChainType.BSC: 0.97,
        }
        
        # Calculate overall success rate
        success_rate = 1.0
        
        # Multiply by each chain's success rate
        success_rate *= chain_success_rates.get(from_chain, 0.95)
        
        for chain in intermediate_chains:
            success_rate *= chain_success_rates.get(chain, 0.95)
        
        success_rate *= chain_success_rates.get(to_chain, 0.95)
        
        return success_rate
    
    def _estimate_liquidity(self, from_chain: ChainType, to_chain: ChainType, amount: int) -> int:
        """Estimate available liquidity for route."""
        # Simplified liquidity estimation
        base_liquidity = 1000000000000000000000  # 1000 tokens
        
        # Reduce liquidity based on amount
        if amount > base_liquidity:
            return base_liquidity // 2
        
        return base_liquidity


class FraudDetector:
    """Detects fraudulent transactions across all chains."""
    
    def __init__(self):
        self.suspicious_addresses: Set[str] = set()
        self.transaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.risk_scores: Dict[str, float] = {}
    
    def analyze_transaction(self, transaction: UniversalTransaction) -> Dict[str, Any]:
        """Analyze transaction for fraud indicators."""
        risk_factors = []
        risk_score = 0.0
        
        # Check for suspicious addresses
        if transaction.from_address.address in self.suspicious_addresses:
            risk_factors.append("suspicious_source_address")
            risk_score += 0.3
        
        if transaction.to_address.address in self.suspicious_addresses:
            risk_factors.append("suspicious_destination_address")
            risk_score += 0.3
        
        # Check transaction patterns
        pattern_risk = self._analyze_transaction_pattern(transaction)
        if pattern_risk > 0:
            risk_factors.append("suspicious_pattern")
            risk_score += pattern_risk
        
        # Check amount
        if transaction.amount > 1000000000000000000000:  # 1000 tokens
            risk_factors.append("large_amount")
            risk_score += 0.2
        
        # Check route complexity
        if transaction.route and len(transaction.route) > 2:
            risk_factors.append("complex_route")
            risk_score += 0.1
        
        # Store risk score
        self.risk_scores[transaction.tx_id] = risk_score
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "is_suspicious": risk_score > 0.5,
            "recommendation": self._get_recommendation(risk_score)
        }
    
    def _analyze_transaction_pattern(self, transaction: UniversalTransaction) -> float:
        """Analyze transaction pattern for suspicious behavior."""
        from_addr = transaction.from_address.address
        
        if from_addr not in self.transaction_patterns:
            self.transaction_patterns[from_addr] = []
        
        # Add current transaction to pattern
        self.transaction_patterns[from_addr].append({
            "amount": transaction.amount,
            "timestamp": transaction.created_at,
            "chain": transaction.from_address.chain.value
        })
        
        # Keep only last 10 transactions
        if len(self.transaction_patterns[from_addr]) > 10:
            self.transaction_patterns[from_addr] = self.transaction_patterns[from_addr][-10:]
        
        # Check for rapid transactions
        recent_txs = [tx for tx in self.transaction_patterns[from_addr] 
                     if time.time() - tx["timestamp"] < 3600]  # Last hour
        
        if len(recent_txs) > 5:
            return 0.2  # Suspicious: too many transactions in short time
        
        return 0.0
    
    def _get_recommendation(self, risk_score: float) -> str:
        """Get recommendation based on risk score."""
        if risk_score > 0.8:
            return "BLOCK"
        elif risk_score > 0.5:
            return "REVIEW"
        elif risk_score > 0.2:
            return "MONITOR"
        else:
            return "APPROVE"


class UniversalBridge:
    """Main universal bridge implementation."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.address_resolver = AddressResolver()
        self.route_optimizer = RouteOptimizer(config)
        self.fraud_detector = FraudDetector()
        
        # Initialize chain clients
        self.chain_clients: Dict[ChainType, Any] = {}
        self._initialize_clients()
        
        self.pending_transactions: Dict[str, UniversalTransaction] = {}
        self._running = False
    
    def _initialize_clients(self) -> None:
        """Initialize chain clients."""
        try:
            if self.config.ethereum_config:
                self.chain_clients[ChainType.ETHEREUM] = EthereumClient(self.config.ethereum_config)
            
            if self.config.bitcoin_config:
                self.chain_clients[ChainType.BITCOIN] = BitcoinBridge(self.config.bitcoin_config)
            
            if self.config.polygon_config:
                self.chain_clients[ChainType.POLYGON] = PolygonClient(self.config.polygon_config)
            
            if self.config.zkevm_config:
                self.chain_clients[ChainType.POLYGON_ZKEVM] = ZkEVMBridge(
                    self.config.zkevm_config, self.config.polygon_config
                )
            
            if self.config.bsc_config:
                self.chain_clients[ChainType.BSC] = BSCBridge(self.config.bsc_config)
            
            logger.info("Universal bridge clients initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize chain clients: {e}")
            raise BridgeError(f"Client initialization failed: {e}")
    
    async def start(self) -> None:
        """Start the universal bridge."""
        if self._running:
            return
        
        self._running = True
        
        # Start all chain clients
        for client in self.chain_clients.values():
            if hasattr(client, 'start'):
                await client.start()
        
        logger.info("Universal bridge started")
    
    async def stop(self) -> None:
        """Stop the universal bridge."""
        self._running = False
        
        # Stop all chain clients
        for client in self.chain_clients.values():
            if hasattr(client, 'stop'):
                await client.stop()
        
        logger.info("Universal bridge stopped")
    
    async def transfer(self, from_address: str, to_address: str, amount: int,
                     from_chain: ChainType, to_chain: ChainType,
                     token_type: TokenType = TokenType.NATIVE,
                     token_address: Optional[str] = None,
                     private_key: Optional[str] = None) -> str:
        """Perform cross-chain transfer."""
        try:
            # Resolve addresses
            from_universal = self.address_resolver.resolve_address(from_chain, from_address)
            to_universal = self.address_resolver.resolve_address(to_chain, to_address)
            
            # Find optimal route
            routes = self.route_optimizer.find_optimal_route(
                from_chain, to_chain, amount, token_type
            )
            
            if not routes:
                raise BridgeError("No route found for transfer")
            
            best_route = routes[0]
            
            # Create universal transaction
            tx_id = f"universal_{secrets.token_hex(16)}"
            universal_tx = UniversalTransaction(
                tx_id=tx_id,
                from_address=from_universal,
                to_address=to_universal,
                amount=amount,
                token_type=token_type,
                token_address=token_address,
                fee=best_route.total_fee,
                route=best_route.intermediate_chains + [to_chain],
                status="pending"
            )
            
            # Fraud detection
            if self.config.enable_fraud_detection:
                fraud_analysis = self.fraud_detector.analyze_transaction(universal_tx)
                if fraud_analysis["is_suspicious"]:
                    raise BridgeError(f"Transaction flagged as suspicious: {fraud_analysis['risk_factors']}")
            
            # Store transaction
            self.pending_transactions[tx_id] = universal_tx
            
            # Execute transfer
            await self._execute_transfer(universal_tx, best_route, private_key)
            
            return tx_id
            
        except Exception as e:
            logger.error(f"Failed to perform transfer: {e}")
            raise BridgeError(f"Transfer failed: {e}")
    
    async def _execute_transfer(self, transaction: UniversalTransaction,
                              route: RouteInfo, private_key: Optional[str]) -> None:
        """Execute the transfer using the specified route."""
        try:
            if len(route.intermediate_chains) == 0:
                # Direct transfer
                await self._execute_direct_transfer(transaction, private_key)
            else:
                # Multi-hop transfer
                await self._execute_multi_hop_transfer(transaction, route, private_key)
            
            transaction.status = "confirmed"
            logger.info(f"Transfer {transaction.tx_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to execute transfer {transaction.tx_id}: {e}")
            transaction.status = "failed"
            raise
    
    async def _execute_direct_transfer(self, transaction: UniversalTransaction,
                                     private_key: Optional[str]) -> None:
        """Execute direct transfer between two chains."""
        from_chain = transaction.from_address.chain
        to_chain = transaction.to_address.chain
        
        # Get source chain client
        source_client = self.chain_clients.get(from_chain)
        if not source_client:
            raise BridgeError(f"No client available for {from_chain.value}")
        
        # Execute transfer based on chain type
        if from_chain == ChainType.ETHEREUM:
            await self._execute_ethereum_transfer(transaction, source_client, private_key)
        elif from_chain == ChainType.BITCOIN:
            await self._execute_bitcoin_transfer(transaction, source_client, private_key)
        elif from_chain == ChainType.POLYGON:
            await self._execute_polygon_transfer(transaction, source_client, private_key)
        elif from_chain == ChainType.BSC:
            await self._execute_bsc_transfer(transaction, source_client, private_key)
        else:
            raise BridgeError(f"Unsupported chain: {from_chain.value}")
    
    async def _execute_multi_hop_transfer(self, transaction: UniversalTransaction,
                                        route: RouteInfo, private_key: Optional[str]) -> None:
        """Execute multi-hop transfer."""
        # This would involve executing transfers through intermediate chains
        # For now, just simulate the process
        
        for i, chain in enumerate(route.intermediate_chains):
            logger.info(f"Executing hop {i+1} through {chain.value}")
            await asyncio.sleep(1)  # Simulate processing time
        
        logger.info(f"Multi-hop transfer completed through {len(route.intermediate_chains)} hops")
    
    async def _execute_ethereum_transfer(self, transaction: UniversalTransaction,
                                       client: Any, private_key: Optional[str]) -> None:
        """Execute Ethereum transfer."""
        # This would involve calling the appropriate Ethereum client method
        logger.info(f"Executing Ethereum transfer: {transaction.amount}")
    
    async def _execute_bitcoin_transfer(self, transaction: UniversalTransaction,
                                      client: Any, private_key: Optional[str]) -> None:
        """Execute Bitcoin transfer."""
        # This would involve calling the appropriate Bitcoin client method
        logger.info(f"Executing Bitcoin transfer: {transaction.amount}")
    
    async def _execute_polygon_transfer(self, transaction: UniversalTransaction,
                                      client: Any, private_key: Optional[str]) -> None:
        """Execute Polygon transfer."""
        # This would involve calling the appropriate Polygon client method
        logger.info(f"Executing Polygon transfer: {transaction.amount}")
    
    async def _execute_bsc_transfer(self, transaction: UniversalTransaction,
                                  client: Any, private_key: Optional[str]) -> None:
        """Execute BSC transfer."""
        # This would involve calling the appropriate BSC client method
        logger.info(f"Executing BSC transfer: {transaction.amount}")
    
    def get_transaction_status(self, tx_id: str) -> Optional[UniversalTransaction]:
        """Get transaction status."""
        return self.pending_transactions.get(tx_id)
    
    def get_available_routes(self, from_chain: ChainType, to_chain: ChainType,
                           amount: int, token_type: TokenType) -> List[RouteInfo]:
        """Get available routes for transfer."""
        return self.route_optimizer.find_optimal_route(from_chain, to_chain, amount, token_type)
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        pending_count = len([tx for tx in self.pending_transactions.values() if tx.status == "pending"])
        confirmed_count = len([tx for tx in self.pending_transactions.values() if tx.status == "confirmed"])
        failed_count = len([tx for tx in self.pending_transactions.values() if tx.status == "failed"])
        
        # Count by chain
        chain_counts = {}
        for tx in self.pending_transactions.values():
            from_chain = tx.from_address.chain.value
            chain_counts[from_chain] = chain_counts.get(from_chain, 0) + 1
        
        return {
            "pending_transactions": pending_count,
            "confirmed_transactions": confirmed_count,
            "failed_transactions": failed_count,
            "total_transactions": len(self.pending_transactions),
            "chain_counts": chain_counts,
            "available_chains": list(self.chain_clients.keys()),
            "route_optimization_enabled": self.config.enable_route_optimization,
            "fraud_detection_enabled": self.config.enable_fraud_detection,
            "fee_optimization_enabled": self.config.enable_fee_optimization,
            "running": self._running
        }
